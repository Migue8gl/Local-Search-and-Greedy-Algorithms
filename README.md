# Local-Search-and-Greedy-Algorithms
Greddy and Local Search techniques for the Weight Learning Problem.

- This practice was done in the Metaheuristic (Metaheurísticas) subject at the UGR. It consists in study, understand and execute local search and greedy algorithms to solve the weight learning problem in three different data sets.

### Data Sets:
  * Parkinsons – data used to distinguish presence or absence
of Parkinson's disease. It consists of 195 examples, 23 attributes and 2 classes.

  * Spectf-heart – contains attributes calculated from images
medical computed tomography. The task is to find out if the
physiology of the heart is correct. Consists of 267 examples, 45 attributes
and 2 classes.

  * Ionosphere – radar data collected at Goose Bay. Its objective is to
classify electrons as "good" and "bad" depending on whether they have
some kind of structure in the ionosphere. It consists of 352 examples, 34
attributes and 2 classes.

### Algorithms:
  * Relief: This algorithm modifies the weights vector from the difference of
calculated distance between own element with its friend (neighbor with same
class) closest and with its enemy (neighbor with a different class) closest.
If the weights are negative we can neglect them.
This algorithm is based on giving more strength to the characteristics that strongly distinguish between two elements of different classes.

  * Local Search: This algorithm is based on an iterative method that starts with an
initial solution, in our case we generate a vector of random weights with
an uniform distribution, and that seeks improvements by making local modifications.
In our case we use "first best", it will be saving the
best solution every time.
The "mutation" that will modify our vector of weights is generated with a normal distribution. The objective function to be maximized
will be F(w) = alpha * class_rate(w) + (1-alpha) * red_rate(w).

  * 1-NN: We will use the classifier with a weights vector w = [1...1], so its main function will be to find the nearest neighbor to each element of the sample.
