

# Jigsaw - Agile Community Rules Classification

## Given Dataset:
- train.csv - the training dataset
    - body - the text of the comment
    - rule - the rule the comment is judged to be in violation of
    - subreddit - the forum the comment was made in
    - positive_example_{1,2} - examples of comments that violate the rule
    - negative_example_{1,2} - examples of comments that do not violate the rule
    - rule_violation - the binary target

- test.csv - the test dataset; objective is to predict the probability of a rule_violation.
    NOTE: The test dataset contains additional rules that are not seen in the in the training data, so models must be flexible to unseen rules.
- sample_submission.csv - a sample submission file in the correct format.

## Evaluation Metric:
- Submissions are evaluated on column-averaged AUC.

-----

## Analysis Reports:
- target is balanced with respect to rule_violation.
- train.csv file is (2029,9) with rule violation ratio of 1031:998
- there are only two rules in the training set, so models must be flexible to unseen rules.

## Baseline:

