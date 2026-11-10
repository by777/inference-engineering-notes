# Where this leaves you · Production

<!-- https://learn-inference.com/chapters/production/closing -->

Three layers, and you now have all of them. The [runtime](https://learn-inference.com/chapters/models), where arithmetic intensity tells you which resource you are short of and the techniques in [Chapter 5](https://learn-inference.com/chapters/techniques) spend one to buy another. The infrastructure, where the problem changes shape every time you grow an order of magnitude. And the tooling, which decides whether any of it is operable by more than one person.

The specifics will age. Hardware generations, engine version numbers, and benchmark figures all have a shelf life measured in months. The constraints underneath move far more slowly: memory bandwidth has bounded decode for years, attention has been quadratic since 2017, and no amount of software makes a byte arrive faster than the bus allows.

The durable skill is finding the bottleneck. Everything else in this book can be re-derived from it.
