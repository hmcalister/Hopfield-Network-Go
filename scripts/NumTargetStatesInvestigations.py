import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

FIGSIZE = (10,8)
trialDataColumns = ["TrialIndex", "NumTargetStates", "NumberStableStates"]
trialData = pd.read_parquet("../data/trialData.pq", columns=trialDataColumns)
stateDataColumns = ['TrialIndex', 'StateIndex', 'Stable', 'NumSteps', "DistancesToLearned"]
stateData = pd.read_parquet("../data/stateData.pq", columns=stateDataColumns)

joinCol = ["TrialIndex"]
df = stateData.join(trialData.set_index(joinCol), on=joinCol)


stableStateData = df.loc[df["Stable"]==True].copy()
stableStateData["MinimumDistanceToLearned"] = (stableStateData["DistancesToLearned"].apply(lambda x: int(np.min(x))))

minDists = stableStateData["MinimumDistanceToLearned"].value_counts()
minDistDf = pd.DataFrame()
minDistDf["MinimumDistanceToLearned"] = np.array(minDists.keys(), dtype=int)
minDistDf["Counts"] = minDists.values

numSteps = stableStateData["NumSteps"].value_counts()
numStepsDf = pd.DataFrame()
numStepsDf["NumSteps"] = np.array(numSteps.keys(), dtype=int)
numStepsDf["Counts"] = numSteps.values


# -------------------------------------------------------------------------------------------------
# Stable States vs Unstable States Count

stableCounts = stateData["Stable"].value_counts()
stableDf = pd.DataFrame()
stableDf["Stable"] = stableCounts.keys()
stableDf["Counts"] = stableCounts.values


fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot()
sns.barplot(stableDf, x="Stable", y="Counts", ax=ax)
plt.title("Count of Stable and Unstable States")
plt.show()

# -------------------------------------------------------------------------------------------------
stabilityRatioByNumTargetStates = trialData.copy()
stabilityRatioByNumTargetStates["StabilityRatio"] = stabilityRatioByNumTargetStates["NumberStableStates"]/1000
stabilityRatioByNumTargetStates = stabilityRatioByNumTargetStates.groupby("NumTargetStates").aggregate({"StabilityRatio": [np.mean, np.std, "count"]})
stabilityRatioByNumTargetStates = stabilityRatioByNumTargetStates.reset_index()
stabilityRatioByNumTargetStates[("StabilityRatio","err")] = \
    1.96 * stabilityRatioByNumTargetStates["StabilityRatio"]["std"]/np.sqrt(stabilityRatioByNumTargetStates["StabilityRatio"]["count"])
fig = plt.figure(figsize=FIGSIZE)
plt.errorbar(x=stabilityRatioByNumTargetStates["NumTargetStates"], 
    y=stabilityRatioByNumTargetStates["StabilityRatio"]["mean"], 
    yerr=stabilityRatioByNumTargetStates["StabilityRatio"]["err"], 
    marker="o",
    ecolor=(0,0,0,0.3),
    capsize=3)
plt.title("Stability Ratio\nby Number of Target States")
plt.xlabel("Number of Target States")
plt.ylabel("Stability Ratio")
plt.show()


# -------------------------------------------------------------------------------------------------
# Minimum Distance to Learned Attractor Distribution (all trials)

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot()
sns.barplot(minDistDf, x="MinimumDistanceToLearned", y="Counts", ax=ax)
plt.title("Distribution of Minimum Distance to Nearest Learned State\nAll Trials")
plt.xticks(rotation=80)
plt.tight_layout()
plt.show()


# -------------------------------------------------------------------------------------------------
# Mean Minimum Distance to Learned Attractor Distribution (grouped by NumTargetStates)

minDistsByNumTargetStates = stableStateData.copy()
minDistsByNumTargetStates = minDistsByNumTargetStates.loc[minDistsByNumTargetStates["MinimumDistanceToLearned"] > 0]
minDistsByNumTargetStates = stableStateData.groupby("NumTargetStates").aggregate({"MinimumDistanceToLearned": [np.mean, np.std, "count"]})
minDistsByNumTargetStates = minDistsByNumTargetStates.reset_index()
minDistsByNumTargetStates[("MinimumDistanceToLearned","err")] = \
    1.96 * minDistsByNumTargetStates["MinimumDistanceToLearned"]["std"]/np.sqrt(minDistsByNumTargetStates["MinimumDistanceToLearned"]["count"])
fig = plt.figure(figsize=FIGSIZE)
plt.errorbar(x=minDistsByNumTargetStates["NumTargetStates"], 
    y=minDistsByNumTargetStates["MinimumDistanceToLearned"]["mean"], 
    yerr=minDistsByNumTargetStates["MinimumDistanceToLearned"]["err"], 
    marker="o",
    ecolor=(0,0,0,0.3),
    capsize=3)
plt.title("Mean Minimum Distance to Nearest Learned Attractor\nby Number of Target States")
plt.xlabel("Number of Target States")
plt.ylabel("Mean Minimum Distance to Nearest Learned Attractor")
plt.show()

# -------------------------------------------------------------------------------------------------
# Number of steps to reach stable state (all trials)
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot()
sns.barplot(numStepsDf, x="NumSteps", y="Counts", ax=ax)
plt.title("Distribution of Number of Steps to Relax\nAll Trials")
plt.xticks(np.arange(4, numStepsDf["NumSteps"].max(), 5))
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------
# Number of steps to reach stable state (by NumNumTargetStates)

numStepsByNumTargetStates = stableStateData.copy()
numStepsByNumTargetStates = stableStateData.groupby("NumTargetStates").aggregate({"NumSteps": [np.mean, np.std, "count"]})
numStepsByNumTargetStates = numStepsByNumTargetStates.reset_index()
numStepsByNumTargetStates[("NumSteps","err")] = \
    1.96 * numStepsByNumTargetStates["NumSteps"]["std"]/np.sqrt(numStepsByNumTargetStates["NumSteps"]["count"])
fig = plt.figure(figsize=FIGSIZE)
plt.errorbar(x=numStepsByNumTargetStates["NumTargetStates"], 
    y=numStepsByNumTargetStates["NumSteps"]["mean"], 
    yerr=numStepsByNumTargetStates["NumSteps"]["err"], 
    marker="o",
    ecolor=(0,0,0,0.7),
    capsize=3)
plt.title("Mean Number of Steps Taken to Relax State\nby Number of Target States")
plt.xlabel("Number of Target States")
plt.ylabel("Mean Number of Steps Taken to Relax State")
plt.show()