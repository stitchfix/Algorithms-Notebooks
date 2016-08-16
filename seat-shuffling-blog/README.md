# seat-shuffling
A Jupyter notebook for facilitating and visualizing randomized office reseating

## This isn't intended to be a fully generic solution
If you'd like to try using our tool, here's how:

1. Fill in the names of people on the team in `team.csv` (replacing things like `First1` and `Last1` with people's actual first and last names).  Specify whether each person needs a seat (`True`) or not (`False`), and what their original seat numbers are.
(Specifying the original seat isn't necessary for shuffling -- it's just needed for visualizing permutation cycles.)
2. If you want to specify some seats by hand, fill those in in `fixed_seats.csv`.
3. In order to use this code for your own purposes, you'll probably need to change the seat numbering and the layout in the visualization code.
 
After customizing the code for your own layout/needs, we hope this code proves useful and fun for you.
