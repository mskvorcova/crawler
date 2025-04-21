package crawler

import (
	"context"
	"crawler/internal/fs"
	"crawler/internal/workerpool"
	"encoding/json"
	"fmt"
	"sync"
)

// Configuration holds the configuration for the crawler, specifying the number of workers for
// file searching, processing, and accumulating tasks. The values for SearchWorkers, FileWorkers,
// and AccumulatorWorkers are critical to efficient performance and must be defined in
// every configuration.
type Configuration struct {
	SearchWorkers      int // Number of workers responsible for searching files.
	FileWorkers        int // Number of workers for processing individual files.
	AccumulatorWorkers int // Number of workers for accumulating results.
}

// Combiner is a function type that defines how to combine two values of type R into a single
// result. Combiner is not required to be thread-safe
//
// Combiner can either:
//   - Modify one of its input arguments to include the result of the other and return it,
//     or
//   - Create a new combined result based on the inputs and return it.
//
// It is assumed that type R has a neutral element (forming a monoid)
type Combiner[R any] func(current R, accum R) R

// Crawler represents a concurrent crawler implementing a map-reduce model with multiple workers
// to manage file processing, transformation, and accumulation tasks. The crawler is designed to
// handle large sets of files efficiently, assuming that all files can fit into memory
// simultaneously.
type Crawler[T, R any] interface {
	// Collect performs the full crawling operation, coordinating with the file system
	// and worker pool to process files and accumulate results. The result type R is assumed
	// to be a monoid, meaning there exists a neutral element for combination, and that
	// R supports an associative combiner operation.
	// The result of this collection process, after all reductions, is returned as type R.
	//
	// Important requirements:
	// 1. Number of workers in the Configuration is mandatory for managing workload efficiently.
	// 2. FileSystem and Accumulator must be thread-safe.
	// 3. Combiner does not need to be thread-safe.
	// 4. If an accumulator or combiner function modifies one of its arguments,
	//    it should return that modified value rather than creating a new one,
	//    or alternatively, it can create and return a new combined result.
	// 5. Context cancellation is respected across workers.
	// 6. Type T is derived by json-deserializing the file contents, and any issues in deserialization
	//    must be handled within the worker.
	// 7. The combiner function will wait for all workers to complete, ensuring no goroutine leaks
	//    occur during the process.
	Collect(
		ctx context.Context,
		fileSystem fs.FileSystem,
		root string,
		conf Configuration,
		accumulator workerpool.Accumulator[T, R],
		combiner Combiner[R],
	) (R, error)
}

type crawlerImpl[T, R any] struct{}

func New[T, R any]() *crawlerImpl[T, R] {
	return &crawlerImpl[T, R]{}
}

// catch is a utility function to recover from panics and convert them into errors.
// If a panic occurs, the provided error pointer is set with the recovered error.
func catch(errorRes *error)  {
	if recoverErr := recover(); recoverErr != nil {
		if errRead, ok := recoverErr.(error); ok {
			*errorRes = errRead
		} else {
			*errorRes = fmt.Errorf("unknown panic: %v", recoverErr)
		}
	}
}

func (c *crawlerImpl[T, R]) Collect(
	ctx context.Context,
	fileSystem fs.FileSystem,
	root string,
	conf Configuration,
	accumulator workerpool.Accumulator[T, R],
	combiner Combiner[R],
) (R, error) {
	// Channel for communicating file paths between the search and transform stages.
	files := make(chan string)

	// Variable to capture errors encountered during the process.
	var errorRes error

	// searcher defines the logic to read a directory and return its contents.
	// - For each directory, it reads its entries using `fileSystem.ReadDir`.
	// - If an entry is a directory, its path is appended to the result slice for further processing.
	// - If an entry is a file, its path is sent to the `files` channel for transformation.
	// - The function respects the `ctx` context and stops sending paths if the context is cancelled.
	// - Errors encountered during directory reading are stored in `errorRes`.
	searcher := func(parent string) []string {
		defer catch(&errorRes) // Ensure any panic is caught and stored as an error.
		dir, err := fileSystem.ReadDir(parent)
		var result []string
		if err != nil {
			errorRes = err // Store error if directory reading fails.
			return nil
		}
		for _, name := range dir {
			path := fileSystem.Join(parent, name.Name())
			if name.IsDir() {
				// Append directories to result for recursive traversal.
				result = append(result, path)
			} else {
				// Send file paths to the channel unless the context is cancelled.
				select {
				case <-ctx.Done():
					// Stop processing if the context is cancelled.
					return nil
				case files <- path:
				}
			}
		}
		return result
	}

	// transformer reads and processes individual files, decoding their contents into type T.
	// - Opens the file specified by `filePath`.
	// - Uses a JSON decoder to deserialize the file contents into a value of type T.
	// - If any error occurs (e.g., file not found or decoding fails), it stores the error in `errorRes`
	//   and returns the zero value of type T.
	// - Ensures that the file handle is closed properly after processing.
	transformer := func(filePath string) T {
		defer catch(&errorRes) // Ensure any panic is caught and stored as an error.
		var zeroValue T // Default zero value of type T.

		// Open the file at the specified path.
		file, err := fileSystem.Open(filePath)
		if err != nil {
			errorRes = err // Store error if file opening fails.
			return zeroValue
		}
		defer file.Close() // Ensure file is closed after processing.

		// Decode the file's contents into a value of type T.
		var data T
		decoder := json.NewDecoder(file)
		if err2 := decoder.Decode(&data); err2 != nil {
			errorRes = err2 // Store error if decoding fails.
			return zeroValue
		}

		return data // Return the successfully decoded value of type T.
	}

	// Create worker pools for transformation and accumulation stages.
	wp := workerpool.New[T, R]()
	swp := workerpool.New[string, T]()

	wg := new(sync.WaitGroup)

	// Goroutine to list files and send them to the `files` channel.
	wg.Add(1)
	go func() {
		defer wg.Done()
		select {
		case <-ctx.Done():
			return
		default:
			swp.List(ctx, conf.SearchWorkers, root, searcher)
			close(files)
		}
	}()

	// Stage for transforming file paths into type T.
	transformStage := swp.Transform(ctx, conf.FileWorkers, files, transformer)
	// Stage for accumulating results into the final result of type R.
	accumulateStage := wp.Accumulate(ctx, conf.AccumulatorWorkers, transformStage, accumulator)

	// Goroutine to combine accumulated results using the combiner function.
	var finalResult R
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <- ctx.Done():
				return
			case el, ok := <-accumulateStage:
				if !ok {
					return
				}
				select {
				case <-ctx.Done():
					return
				default:
					finalResult = combiner(finalResult, el)
				}
			}
		}
	}()

	// Wait for all goroutines to complete.
	wg.Wait()

	// Return the final result and any errors encountered during execution.
	if errorRes != nil {
		return finalResult, errorRes
	}

	return finalResult, ctx.Err()
}
