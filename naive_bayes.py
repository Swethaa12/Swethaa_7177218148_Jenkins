import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('nb.csv')

op = dict(df['PlayGolf'].value_counts())
rows = df.shape[0]

def find_probs(out_label):
    prob = {}
    for column in df.columns[:-1]:
        for cat in df[column].unique():
            temp = {}
            for output in df[out_label].unique():
                occurrences = ((df[column] == cat) & (df[out_label] == output)).sum()
                temp[output] = occurrences / op[output]
            prob[str(cat)] = temp
    prob_df = pd.DataFrame(prob)
    return prob_df

def predict(X, prob_df):
    p_yes = op['Yes'] / rows
    p_no = op['No'] / rows
    for attr in X:
        p_yes *= prob_df[attr]['Yes']
        p_no *= prob_df[attr]['No']
    P_yes = round(p_yes / (p_yes + p_no), 2)
    P_no = round(p_no / (p_yes + p_no), 2)
    return "Yes" if P_yes >= P_no else "No"

def display_performance_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='binary', pos_label='Yes')
    recall = recall_score(true_labels, predicted_labels, average='binary', pos_label='Yes')
    f1 = f1_score(true_labels, predicted_labels, average='binary', pos_label='Yes')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

prob_df = find_probs('PlayGolf')
user_input = {
    'Outlook': input("Enter Outlook (Sunny, Overcast, Rainy): "),
    'Temperature': input("Enter Temperature (Hot, Mild, Cool): "),
    'Humidity': input("Enter Humidity (High, Normal): "),
    'Wind': input("Enter Wind (TRUE, FALSE): ")
}
prediction = predict([user_input[attr] for attr in df.columns[:-1]], prob_df)
print("Prediction:", prediction)
true_labels = df['PlayGolf']
predictions = [predict(row[:-1], prob_df) for index, row in df.iterrows()]
display_performance_metrics(true_labels, predictions)

print(prob_df)
