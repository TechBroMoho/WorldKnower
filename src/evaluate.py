from pathlib import Path # credit folders and specify file locations
import matplotlib.pyplot as plt # create graphs
import torch #used to run the trained AI model

from sklearn.metrics import (accuracy_score,  # percentage of correct guesses
                             f1_score,        # Measures prediction quality using precision and recall
                             confusion_matrix,# Counts how many predictions were correct/incorrect for each country
                             classification_report, # produces a detailed report for every country
                             ConfusionMatrixDisplay) # turns the confusion matrix into a graph


def collect_predictions(model, val_loader): # function that takes a trained model and validation data loader
    y_true = [] # creates an empty list to store correct country labels
    y_pred = [] # creates an empty list to store the model's predicted country labels


    model.eval() # switching the model to evaluation mode

    with torch.no_grade(): #disabeling the graident calculations because we aren't training
        for images, labels in val_loader: # go through validation images one batch at a time
            outputs = model(images) # feed the images into the model to obtain country scores
            predictions = outputs.argmax(dim=1) #selecting the ountry with the highest score for each image

            y_true.extend(labels.cpu().tolist()) #convert the correct labels to a python list
            y_pred.extend(predictions.cpu().tolist()) #convert the predictions to a python list
    return y_true, y_pred


def test_evaluation(): 
    y_true = ["Japan", "South Korea", "Ecuador", "Urugauy"]
    y_pred = ["Japan", "South Korea", "Ecuador", "Japan"]

    accuracy, macro_f1 = evaluate(y_true, y_pred) 

    assert accuracy == 0.75
    assert abs(macro_f1-2/3) < 0.0001

    print("All tests passed!")

def evaluate(y_true, y_pred):# function to compare correct labels and predictions 
    accuracy = accuracy_score(y_true, y_pred) #calculate the accuracy by comparing actual and predicted countries 

    macro_f1 = f1_score( #calculating the macro-averaged F1 score
        y_true, 
        y_pred,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy: {accuracy:.2%}") #display the accuracy as percentage
    print(f"Macro F1: {macro_f1:.4f}") #display the macro F1

    cm = confusion_matrix(y_true, y_pred) #calculate the confusion matrix

    report = classification_report( #generate a detailed classification report
        y_true,
        y_pred,
        zero_division=0
    )

    print("\nClassification Report:") 
    print(report) #print classification report

    display = ConfusionMatrixDisplay.from_predictions( #generate a confusion matrix graph from the predictions
        y_true,
        y_pred,
        cmap="Blues",
        xticks_rotation=45
    )

    output_dir = Path("results/figures") #specify where the graph will be saved
    output_dir.mkdir(parents=True, exist_ok=True) #creates new folder

    plt.tight_layout()
    plt.savefig( #saving the confusion matrix as a PNG image
        output_dir / "confusion_matrix.png", # output file location
        bbox_inches="tight"
    )
    plt.close()

    print("Confusion matrix saved!")


    return accuracy, macro_f1 #returnng the accuracy and macro F1


if __name__ == "__main__": #running only when filed is executed directly
    test_evaluation() # run to verify evaluation works

def run_real_evaluation(model, val_loader): #function used to evalate a teammate' trained model
    y_true, y_pred = collect_predictions(model, val_loader)

    accuracy, macro_f1 = evaluate(y_true, y_pred)

    return accuracy, macro_f1
