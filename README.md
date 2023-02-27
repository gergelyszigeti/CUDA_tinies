argmax of hugevector:
It is a recreation of my good old "which thread has the largest value" GPU kernel.
I wrote a quite similar function for a GPU RANSAC algorithm (plane fitting on a point cloud).
As far as I remember that was always used for 1024 values (random plane candidates).
Now I'm planning to extend it to way more values.

In current state, this is really a recreation of the old thing, the only
difference that this new version can also handle less than 1024 values.
However, the number of values should still be divisible by 32.
Update: Now it is working with several millions of values also.
TODO: put it into a separate function, which could also have the final GPU-CPU copy of the results. 

