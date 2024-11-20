Iris Flower Classification Program
Overview
This program provides an interactive tool to classify Iris flowers into one of three species: Setosa, Versicolor, or Virginica. Using the Random Forest classification algorithm, users can train the model, test its accuracy, and predict the species of a new sample based on user-provided measurements.

The Iris dataset, a classic dataset in machine learning, is used for this purpose. The four input features are:

Sepal Length (cm)
Sepal Width (cm)
Petal Length (cm)
Petal Width (cm)
The target labels correspond to the three Iris species.

Implementation
Libraries Used:

scikit-learn for the machine learning model and dataset handling
numpy for data representation
Python standard libraries for interactive user input
Features of the Program:

Train a Random Forest model on the Iris dataset.
Evaluate the model’s accuracy on a test dataset.
Allow users to input sample measurements to classify a new Iris flower.

Real-World Applications
Botanical Studies:
The model simulates a practical application where botanists could classify plant species based on measurable features. Automating classification reduces manual effort and improves accuracy in species identification.

Quality Control in Agriculture:
Models like this can be adapted to classify agricultural products (e.g., grains, fruits) based on measurable attributes, ensuring consistent quality in production and distribution.

Medical Research:
In the medical field, similar models can classify conditions or diseases based on diagnostic measurements, aiding in faster and more accurate diagnoses.