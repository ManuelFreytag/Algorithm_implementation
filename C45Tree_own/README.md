# Tree learner

This algorithm implementation should only be interpreted as proof of concept.

## Included aspects:
- Supports numeric and discrete data
- Supports multiple label values
- Supports multiple splitting criterion
- Supports missing values
- Supports the command structure of sklearn (fit, predict)

## Critique and missing aspects
- Missing branch prunning
- Missing value handling is implemented with KNN imputation not per native inclusion
- Dependency on pandas and sklearn
- Speed of the learner is not comparable to state of the art implementations (Used datastructures and design choices).

## Running
Add the repository into your library folder. You then use it as you would use every other library ``import C45Tree``
