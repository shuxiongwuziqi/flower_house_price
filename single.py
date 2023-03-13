from sklearn.linear_model import LogisticRegression
import utils

(X_train, y_train), (X_test, y_test) = utils.load_data()

model = LogisticRegression(max_iter=100000, multi_class="multinomial")
# utils.set_initial_params(model)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(accuracy)
