# Sieve

This is a program to perform the [Sieve of Eratosthenes][soe].

## Details

- It's written in C++
- It's working on a ~~`std::map`~~ `std::vector` (faster than map, less memory consumption)
- It's working bitwise on a `long long`
- It's possible to save the current sieve data.
- It's possible to load the saved sieve data.

## Done

- Check code for fails using CPP Check.
- Add feature to stop process.
- Add feature to save current state.
- Add feature to resume aborted process.
- Enable UI for load/save current prime states.
- Enable user to set upper limit for determining prime numbers.
- Enable user to start/stop prime sieving.
- Add feature to export primes.
- Split marking of multiples of primes into threads:
  - [x] Detect number of CPU cores.
  - [x] Split range of sieve into several parts.
  - [x] Develop algorithm to mark multiples of primes in each range.

## Todo

- Features/ideas of the future, e. g. another sieve algorithm, for example the [Sieve of Atkin][soa]
- Find an algorithm for sieving on GPU using [CUDA][cuda]
- Refactor code
  - To have interfaces for
    - [ ] storage
    - [ ] sieving algorithm
  - To inject classes based on new interfaces
  - To have [composition over inheritance][coi]

[coi]: https://en.wikipedia.org/wiki/Composition_over_inheritance
[cuda]: https://en.wikipedia.org/wiki/CUDA
[soa]: https://en.wikipedia.org/wiki/Sieve_of_Atkin
[soe]: https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
