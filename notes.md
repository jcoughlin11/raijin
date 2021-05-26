# Brainstorming

Goal:

```bash
raijin train params.yaml
raijin test params.yaml
raijin setup
raijin analyze /path/to/data
raijin upload /path/to/data
raijin package /path/to/data
```

## Parameter File
* A yaml file containing data for the run (architecture to use, game to play, memory type, memory buffer size, etc.)
* The file should be broken up into sections (one for each of the primary objects/operations in the code)
* Question: Should every parameter be in the file even if it's not being used? For example, if we're not using fixed-Q or double-Q, do we need `updateFreq` in the parameter file since it isn't used? What about parameters specific to pytorch objects? Should they be in there, or just trust the user to add them if needed?

## Train Command
* Trains an agent to play a particular game
* Want to be able to continue training if it ends early
    * Case 1: The game env is deterministic (can stop mid-episode)
    * Case 2: It isn't (can't stop mid-episode since we won't be able to get back to the state we ended at). This means the best we can do is discard everything from an in-progress episode and, when restarting, load the state of things from the end of the previous episode
* An option to cause training to end early (like gadget has)
* Should have callback support
* Should have good support for handling a dynamic number of metrics

## Test Command
* Has the agent play one episode of the game to see how it does
* An option to visualize or not would be cool

## Setup Command
* Used to interactively build a parameter file or new network architecture

## Analyze Command
* Used to create figures of network architectures and plots of various metrics

## Upload Command
* Used to put data products on a place like wholetale or zenodo or wherever (not sure how useful this is, so it might not be needed)

## Package Command
* Creates a tar archive of the data and/or code for sharing or uploading (not sure how useful this actually is, so it might not be needed)

* Random thought: the current git hash of raijin should be included in the output products somewhere so that results can be reproduced exactly

## Design Philosophy
* The code MUST BE CLEAN
* It MUST BE WELL-DOCUMENTED
* It MUST BE WELL-COMMENTED
* It MUST BE EASY TO READ (even at the expense of efficiency)
