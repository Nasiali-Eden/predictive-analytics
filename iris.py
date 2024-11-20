# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
is_model_trained = False  # Flag to track if the model is trained

# Main interactive menu for user input
def main_menu():
    while True:
        print("\n--- Iris Classification Program ---")
        print("1. Train the Model")
        print("2. Test the Model Accuracy")
        print("3. Predict a Sample")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            train_model()
        elif choice == '2':
            test_accuracy()
        elif choice == '3':
            predict_sample()
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

# Function to train the model
def train_model():
    global is_model_trained
    model.fit(X_train, y_train)
    is_model_trained = True
    print("Model has been trained successfully!")

# Function to test the model accuracy
def test_accuracy():
    if not is_model_trained:
        print("Model is not trained yet. Please train the model first.")
        return
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to make predictions on new samples
def predict_sample():
    if not is_model_trained:
        print("Model is not trained yet. Please train the model first.")
        return
    
    print("\nEnter values for a new Iris flower sample.")
    print("\nGuidelines for input values:")
    print("  - Sepal Length: 4.3 to 7.9 cm")
    print("  - Sepal Width: 2.0 to 4.4 cm")
    print("  - Petal Length: 1.0 to 6.9 cm")
    print("  - Petal Width: 0.1 to 2.5 cm")
    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))
        
        # Range check for realistic Iris flower dimensions
        if not (4.3 <= sepal_length <= 7.9 and 2.0 <= sepal_width <= 4.4 and
                1.0 <= petal_length <= 6.9 and 0.1 <= petal_width <= 2.5):
            print("Values are out of the expected range for Iris flower dimensions.")
            return
        
        sample = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(sample)
        print(f"Predicted Iris Class: {data.target_names[prediction][0]}")
    
    except ValueError:
        print("Invalid input. Please enter numerical values.")

# Run the main menu
main_menu()
