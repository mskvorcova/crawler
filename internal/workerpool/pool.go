package workerpool

import (
	"context"
	"sync"
)

// Accumulator is a function type used to aggregate values of type T into a result of type R.
// It must be thread-safe, as multiple goroutines will access the accumulator function concurrently.
// Each worker will produce intermediate results, which are combined with an initial or
// accumulated value.
type Accumulator[T, R any] func(current T, accum R) R

// Transformer is a function type used to transform an element of type T to another type R.
// The function is invoked concurrently by multiple workers, and therefore must be thread-safe
// to ensure data integrity when accessed across multiple goroutines.
// Each worker independently applies the transformer to its own subset of data, and although
// no shared state is expected, the transformer must handle any internal state in a thread-safe
// manner if present.
type Transformer[T, R any] func(current T) R

// Searcher is a function type for exploring data in a hierarchical manner.
// Each call to Searcher takes a parent element of type T and returns a slice of T representing
// its child elements. Since multiple goroutines may call Searcher concurrently, it must be
// thread-safe to ensure consistent results during recursive  exploration.
//
// Important considerations:
//  1. Searcher should be designed to avoid race conditions, particularly if it captures external
//     variables in closures.
//  2. The calling function must handle any state or values in closures, ensuring that
//     captured variables remain consistent throughout recursive or hierarchical search paths.
type Searcher[T any] func(parent T) []T

// Pool is the primary interface for managing worker pools, with support for three main
// operations: Transform, Accumulate, and List. Each operation takes an input channel, applies
// a transformation, accumulation, or list expansion, and returns the respective output.
type Pool[T, R any] interface {
	// Transform applies a transformer function to each item received from the input channel,
	// with results sent to the output channel. Transform operates concurrently, utilizing the
	// specified number of workers. The number of workers must be explicitly defined in the
	// configuration for this function to handle expected workloads effectively.
	// Since multiple workers may call the transformer function concurrently, it must be
	// thread-safe to prevent race conditions or unexpected results when handling shared or
	// internal state. Each worker independently applies the transformer function to its own
	// data subset.
	Transform(ctx context.Context, workers int, input <-chan T, transformer Transformer[T, R]) <-chan R

	// Accumulate applies an accumulator function to the items received from the input channel,
	// with results accumulated and sent to the output channel. The accumulator function must
	// be thread-safe, as multiple workers concurrently update the accumulated result.
	// The output channel will contain intermediate accumulated results as R
	Accumulate(ctx context.Context, workers int, input <-chan T, accumulator Accumulator[T, R]) <-chan R

	// List expands elements based on a searcher function, starting
	// from the given element. The searcher function finds child elements for each parent,
	// allowing exploration in a tree-like structure.
	// The number of workers should be configured based on the workload, ensuring each worker
	// independently processes assigned elements.
	List(ctx context.Context, workers int, start T, searcher Searcher[T])
}

type poolImpl[T, R any] struct{}

func New[T, R any]() *poolImpl[T, R] {
	return &poolImpl[T, R]{}
}

func (p *poolImpl[T, R]) Accumulate(
	ctx context.Context,
	workers int,
	input <-chan T,
	accumulator Accumulator[T, R],
) <-chan R {
	// Channel to return accumulated results.
	result := make(chan R)
	// WaitGroup to track completion of all worker goroutines.
	wg := new(sync.WaitGroup)
	// Slice to store intermediate accumulation results for each worker.
	accumulated := make([]R, workers)
	// Launch worker goroutines to process input and accumulate results.
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(index int) {
			// Mark this worker as done when it exits.
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					// Exit if the context is cancelled.
					return
				case v, ok := <-input:
					// Exit if the input channel is closed.
					if !ok {
						return
					}
					select {
					// Exit if the context is cancelled during processing.
					case <-ctx.Done():
						return
					default:
						// Update the intermediate accumulated value for this worker.
						accumulated[index] = accumulator(v, accumulated[index])
					}
				}
			}
		}(i)
	}
	// Launch a goroutine to collect results from all workers after they complete.
	go func() {
		// Wait for all workers to finish.
		wg.Wait()
		for _, res := range accumulated {
			select {
			// Exit if the context is cancelled during result collection.
			case <-ctx.Done():
				return
			case result <- res:
				// Send the accumulated result to the output channel.
			}
		}
		// Close the result channel when done.
		close(result)
	}()
	// Return the channel with accumulated results.
	return result
}

func (p *poolImpl[T, R]) List(ctx context.Context, workers int, start T, searcher Searcher[T]) {
	// Variables to hold the current and future levels of the tree-like structure.
	var (
		current []T // Current level elements to be processed.
		future  []T // Future level elements to be collected.
		mu      sync.Mutex // Mutex to protect access to the `future` slice.
		wg      sync.WaitGroup // WaitGroup to track completion of all worker goroutines.
	)
	// Initialize the current level with the starting element.
	current = append(current, start)

	// Process levels of the tree until there are no more elements.
	for len(current) > 0 {
		// Pointer to the next task to process.
		nextIndex := 0

		// Launch workers to process tasks dynamically.
		for i := 0; i < workers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done() // Mark this worker as done when it exits.
				for {
					var task T
					mu.Lock()
					if nextIndex < len(current) {
						// Extract the next task and move the pointer.
						task = current[nextIndex]
						nextIndex++
					} else {
						// No more tasks to process.
						mu.Unlock()
						return
					}
					mu.Unlock()

					// Check if the context is canceled before processing the task.
					select {
					case <-ctx.Done():
						return
					default:
						// Process the task and find child elements.
						children := searcher(task)

						// Safely append children to the future slice.
						mu.Lock()
						future = append(future, children...)
						mu.Unlock()
					}
				}
			}()
		}

		// Wait for all workers to finish processing the current level.
		wg.Wait()

		// Move to the next level.
		current = future
		// Clear the future level for reuse.
		future = nil
	}
}

func (p *poolImpl[T, R]) Transform(
	ctx context.Context,
	workers int,
	input <-chan T,
	transformer Transformer[T, R],
) <-chan R {
	// Channel to return transformed results.
	result := make(chan R)
	// WaitGroup to track completion of all worker goroutines.
	wg := new(sync.WaitGroup)
	// Launch worker goroutines to process input and apply the transformer function.
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			// Mark this worker as done when it exits.
			defer wg.Done()
			for {
				select {
				// Exit if the context is cancelled.
				case <-ctx.Done():
					return
				case v, ok := <-input:
					// Exit if the input channel is closed.
					if !ok {
						return
					}
					select {
					// Exit if the context is cancelled during processing.
					case <-ctx.Done():
						return
					// Apply the transformer function and send the result to the output channel.
					case result <- transformer(v):
					}
				}
			}
		}()
	}
	// Launch a goroutine to close the result channel after all workers finish.
	go func() {
		// Wait for all workers to complete.
		wg.Wait()
		// Close the result channel when done.
		close(result)
	}()

	// Return the channel with transformed results.
	return result
}
