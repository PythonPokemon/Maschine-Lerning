import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Beispiel-Daten generieren
np.random.seed(0)
matches = 50
germany_goals = np.random.randint(0, 5, matches)
france_goals = np.random.randint(0, 5, matches)

# Ergebnisse als DataFrame speichern
data = pd.DataFrame({'GermanyGoals': germany_goals, 'FranceGoals': france_goals})

# Trainingsdaten
X = data[['GermanyGoals']]
y = data['FranceGoals']

# Aufteilen der Daten in Trainings- und Testdaten
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineares Regressionsmodell
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_predictions)

# Decision Tree Regressor
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_predictions)

# Random Forest Regressor
forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)
forest_mse = mean_squared_error(y_test, forest_predictions)

# Vorhersage für ein neues Spiel
new_match = pd.DataFrame({'GermanyGoals': [2]})
predicted_france_goals_linear = linear_model.predict(new_match)
predicted_france_goals_tree = tree_model.predict(new_match)
predicted_france_goals_forest = forest_model.predict(new_match)

print(f"Lineares Regressionsmodell: Vorhersage für Deutschland {new_match['GermanyGoals'][0]} - "
      f"{predicted_france_goals_linear[0]:.2f} Tore für Frankreich")
print(f"Decision Tree Regressor: Vorhersage für Deutschland {new_match['GermanyGoals'][0]} - "
      f"{predicted_france_goals_tree[0]:.2f} Tore für Frankreich")
print(f"Random Forest Regressor: Vorhersage für Deutschland {new_match['GermanyGoals'][0]} - "
      f"{predicted_france_goals_forest[0]:.2f} Tore für Frankreich")

# Ausgabe der Mean Squared Errors (MSE) für die Modelle
print(f"Lineares Regressionsmodell MSE: {linear_mse:.2f}")
print(f"Decision Tree Regressor MSE: {tree_mse:.2f}")
print(f"Random Forest Regressor MSE: {forest_mse:.2f}")
