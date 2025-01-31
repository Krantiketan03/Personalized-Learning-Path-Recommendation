import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Sample dataset of subjects with prerequisites and difficulty levels
data = {
    'Subject': ['Machine Learning', 'Data Science', 'Deep Learning', 'Python Programming', 'Statistics', 'Linear Algebra'],
    'Prerequisites': ['Python, Linear Algebra', 'Statistics, Python', 'Machine Learning', 'None', 'None', 'None'],
    'Difficulty': [4, 3, 5, 2, 3, 3],
    'Resources': [
        'https://www.coursera.org/learn/machine-learning',
        'https://www.kaggle.com/learn/data-science',
        'https://www.udacity.com/course/deep-learning-nanodegree--nd101',
        'https://www.learnpython.org/',
        'https://www.khanacademy.org/math/statistics-probability',
        'https://www.khanacademy.org/math/linear-algebra'
    ]
}

df = pd.DataFrame(data)

# K-Nearest Neighbors model for topic recommendation based on difficulty
knn = NearestNeighbors(n_neighbors=2, metric='euclidean')
X = df[['Difficulty']].values
knn.fit(X)

# Function to generate a personalized learning path
def generate_learning_path(subject):
    if subject not in df['Subject'].values:
        return "Subject not found in database."
    
    topic_info = df[df['Subject'] == subject].iloc[0]
    prerequisites = topic_info['Prerequisites']
    difficulty = topic_info['Difficulty']
    resources = topic_info['Resources']
    
    distances, indices = knn.kneighbors([[difficulty]])
    similar_topics = [df['Subject'][i] for i in indices[0] if df['Subject'][i] != subject]
    
    roadmap = {
        "Prerequisites": prerequisites,
        "Main Subject": subject,
        "Suggested Subtopics": similar_topics,
        "Resources": resources
    }
    
    return roadmap

# Example usage
user_input = "Machine Learning"
learning_path = generate_learning_path(user_input)
print("Personalized Learning Path:", learning_path)
