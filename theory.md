### Logistic Regression Implementation in Python

Here’s how you can implement Logistic Regression from scratch using Python, focusing on the sigmoid function and the cost function:


### Explanation of Key Concepts

1. **Sigmoid Function:**
   - **Formula:** \( \sigma(z) = \frac{1}{1 + e^{-z}} \)
   - **Description:** The sigmoid function maps any real-valued number into the range [0, 1]. It's used to predict probabilities in logistic regression.

2. **Sigmoid Curve:**
   - **Description:** The sigmoid curve is an S-shaped curve that represents the probability of the target variable given the features. It shows a smooth transition from 0 to 1.

3. **Euler’s Number (\( e \)):**
   - **Value:** Approximately 2.71828.
   - **Description:** Euler’s number is the base of the natural logarithm. It is used in the sigmoid function to ensure the output is bounded between 0 and 1.

4. **Logarithm:**
   - **Description:** The logarithm function is the inverse of exponentiation. In logistic regression, the log function is used in the cost function to measure the difference between predicted and actual values.

5. **Probability in Logistic Regression:**
   - **Description:** Logistic regression predicts probabilities because the target variable is categorical (e.g., binary outcomes). The sigmoid function ensures the output is between 0 and 1, representing the probability of the positive class.

6. **Logit Function:**
   - **Formula:** \( \text{logit}(p) = \log \left( \frac{p}{1-p} \right) \)
   - **Description:** The logit function is the natural logarithm of the odds of the probability \( p \). In logistic regression, it relates the linear combination of features to the probability of the target class.

In summary, logistic regression uses the sigmoid function to convert a linear combination of features into a probability. The logit function is used to model this relationship, and the cost function (which includes the logarithm) helps in finding the best-fit parameters for the model.
