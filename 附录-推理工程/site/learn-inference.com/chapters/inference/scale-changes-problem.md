# Scale changes the problem · Inference

<!-- https://learn-inference.com/chapters/inference/scale-changes-problem -->

Here is the part that catches teams out. The dominant inference problem is not fixed. It changes as you grow, and each new version of it is largely unrelated to the one you just finished solving.

A team that has spent six months becoming excellent at runtime tuning does not thereby become good at capacity planning. The skills barely overlap. What was a config file becomes a conversation with a vendor about what is physically available in a given region next quarter.

> Figure. What breaks next, by fleet size. Each stage has a different dominant problem, and solving one does not prepare you for the next. The work at the top of the ladder barely resembles the work at the bottom.Illustrative numbers

The last stage is the interesting one. Once workloads are spread across regions and providers, the failure mode stops being “we cannot get GPUs” and becomes “we have GPUs and cannot use them.” One cluster queues requests while another idles. The work at that point is unification: making everything you rent, everywhere, behave as a single pool of compute.

Spreading out has two side benefits worth naming. It protects you from any one region or provider having a bad day, and for a global product it puts inference physically closer to users, which shortens every round trip.
