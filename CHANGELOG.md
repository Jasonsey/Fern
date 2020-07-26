# CHANGE LOG

## Version 0.5

- 0.5.0 (2020-07-26)
  - New Feature:
    - Add FernBalance class for dataset balance
    - Add gpu finder to find which gpu has the largest free memory

## Version 0.4

- 0.4.1 (2020-07-22)
  - New Feature:
    - Stop words use regular matching
    - The `#` comment rule is allowed in word library
    
- 0.4.0 (2020-07-21)
  - New Feature:
    - Add optimized TextCNN model as a built-in model

## Version 0.3

- 0.3.3 (2020-07-09)
  - New Feature:
    - Add predict function for FernModel
    
- 0.3.2 (2020-07-06)
  - Style:
    - Rename BaseTrainer -> FernTrainer
  - Bug Fix:
    - Fix trainer not fitting for multi output labels
 
- 0.3.1 (2020-07-02)
  - Bug Fix:
    - Fix no user words file exist issue
    - Fix cleaned label value should be a list issue
  
- 0.3.0 (2020-07-02)
  - New Feature:
    - using cut_func to split sequence to word list
    - Add save_function for every data precessing stage
    - Mark FernTransformer.output_shape as optional
    - Enable data frame default index
    - Add Sequence2Words class as cut_func generator
    - Use fern as default data frame 

## Version 0.2

- 0.2.0 (2020-05-24)
  - New Features:
    - Modify output shape from list into dict
    - Add label path for storing label data
    - Modify transformed label from array into dictionary
    - Update transformed label and data function
  - Style
    - Change variable name input_col to data_col
    - Change variable name output_col to label_col

## Version 0.1

- 0.1.4 (2020-05-20)
  - Style
    - Change the random state location where it's definded

- 0.1.3 (2020-05-20)
  - New Features
    - add random state for data splitter

- 0.1.2 (2020-05-07)
  - Bug fix
    - Fix training loop quick exit bug

- 0.1.1 (2020-05-03)

  - New features

    - Add github python publish action

- 0.1.0 (2020-05-03)

  - New features

    - Data preprocessing module: data downloader, data cleaner, data transformer and data splitter

    - Model template: model builder, saver, loader and structure printer

    - Model trainer

    - Custom layers: Conv1DPassMask, FlattenPassMask, DenseWithMask, AttentionLayer and ScaledDotProductAttention

    - Custom metrics: BinaryCategoricalAccuarcy

    - Other tools: logging and progress bar

  - Documentation changes

    - Add README 
    - Add README_ZH
    
      

