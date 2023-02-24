import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

FIGSIZE = (10,8)
trialDataColumns = ["TrialIndex", "UnitsUpdated", "NumberStableStates"]
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

stableCount = stateData["Stable"].value_counts()
stableDf = pd.DataFrame()
stableDf["Stable"] = ["Stable", "Unstable"]
stableDf["Counts"] = stableCount.values


fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot()
sns.barplot(stableDf, x="Stable", y="Counts", ax=ax)
plt.title("Count of Stable and Unstable States")
plt.show()

# -------------------------------------------------------------------------------------------------
stabilityRatioByUnitsUpdated = trialData.copy()
stabilityRatioByUnitsUpdated["StabilityRatio"] = stabilityRatioByUnitsUpdated["NumberStableStates"]/1000
stabilityRatioByUnitsUpdated = stabilityRatioByUnitsUpdated.groupby("UnitsUpdated").aggregate({"StabilityRatio": [np.mean, np.std, "count"]})
stabilityRatioByUnitsUpdated = stabilityRatioByUnitsUpdated.reset_index()
stabilityRatioByUnitsUpdated[("StabilityRatio","err")] = \
    1.96 * stabilityRatioByUnitsUpdated["StabilityRatio"]["std"]/np.sqrt(stabilityRatioByUnitsUpdated["StabilityRatio"]["count"])
fig = plt.figure(figsize=FIGSIZE)
plt.errorbar(x=stabilityRatioByUnitsUpdated["UnitsUpdated"], 
    y=stabilityRatioByUnitsUpdated["StabilityRatio"]["mean"], 
    yerr=stabilityRatioByUnitsUpdated["StabilityRatio"]["err"], 
    marker="o",
    ecolor=(0,0,0,0.3),
    capsize=3)
plt.title("Stability Ratio\nby Number of Units Updated")
plt.xlabel("Number of Units Updated (Per Network Step)")
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
# Mean Minimum Distance to Learned Attractor Distribution (grouped by UnitsUpdated)

minDistsByUnitsUpdated = stableStateData.copy()
minDistsByUnitsUpdated = minDistsByUnitsUpdated.loc[minDistsByUnitsUpdated["MinimumDistanceToLearned"] > 0]
minDistsByUnitsUpdated = stableStateData.groupby("UnitsUpdated").aggregate({"MinimumDistanceToLearned": [np.mean, np.std, "count"]})
minDistsByUnitsUpdated = minDistsByUnitsUpdated.reset_index()
minDistsByUnitsUpdated[("MinimumDistanceToLearned","err")] = \
    1.96 * minDistsByUnitsUpdated["MinimumDistanceToLearned"]["std"]/np.sqrt(minDistsByUnitsUpdated["MinimumDistanceToLearned"]["count"])
fig = plt.figure(figsize=FIGSIZE)
plt.errorbar(x=minDistsByUnitsUpdated["UnitsUpdated"], 
    y=minDistsByUnitsUpdated["MinimumDistanceToLearned"]["mean"], 
    yerr=minDistsByUnitsUpdated["MinimumDistanceToLearned"]["err"], 
    marker="o",
    ecolor=(0,0,0,0.3),
    capsize=3)
plt.title("Mean Minimum Distance to Nearest Learned Attractor\nby Number of Units Updated")
plt.xlabel("Number of Units Updated (Per Network Step)")
plt.ylabel("Mean Minimum Distance to Nearest Learned Attractor")
plt.show()

# -------------------------------------------------------------------------------------------------
# Mean Minimum Distance to Learned Attractor Distribution (grouped by UnitsUpdated)

minDistsByUnitsUpdated = stableStateData.copy()
minDistsByUnitsUpdated = minDistsByUnitsUpdated.loc[minDistsByUnitsUpdated["MinimumDistanceToLearned"] == 0]
minDistsByUnitsUpdated = stableStateData.groupby("UnitsUpdated").aggregate({"MinimumDistanceToLearned": [np.mean, np.std, "count"]})
minDistsByUnitsUpdated = minDistsByUnitsUpdated.reset_index()
minDistsByUnitsUpdated[("MinimumDistanceToLearned","err")] = \
    1.96 * minDistsByUnitsUpdated["MinimumDistanceToLearned"]["std"]/np.sqrt(minDistsByUnitsUpdated["MinimumDistanceToLearned"]["count"])
fig = plt.figure(figsize=FIGSIZE)
plt.errorbar(x=minDistsByUnitsUpdated["UnitsUpdated"], 
    y=minDistsByUnitsUpdated["MinimumDistanceToLearned"]["mean"], 
    yerr=minDistsByUnitsUpdated["MinimumDistanceToLearned"]["err"], 
    marker="o",
    ecolor=(0,0,0,0.3),
    capsize=3)
plt.title("Mean Minimum Distance to Nearest Learned Attractor\nby Number of Units Updated")
plt.xlabel("Number of Units Updated (Per Network Step)")
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
# Number of steps to reach stable state (by NumUnitsUpdated)

numStepsByUnitsUpdated = stableStateData.copy()
numStepsByUnitsUpdated = stableStateData.groupby("UnitsUpdated").aggregate({"NumSteps": [np.mean, np.std, "count"]})
numStepsByUnitsUpdated = numStepsByUnitsUpdated.reset_index()
numStepsByUnitsUpdated[("NumSteps","err")] = \
    1.96 * numStepsByUnitsUpdated["NumSteps"]["std"]/np.sqrt(numStepsByUnitsUpdated["NumSteps"]["count"])
fig = plt.figure(figsize=FIGSIZE)
plt.errorbar(x=numStepsByUnitsUpdated["UnitsUpdated"], 
    y=numStepsByUnitsUpdated["NumSteps"]["mean"], 
    yerr=numStepsByUnitsUpdated["NumSteps"]["err"], 
    marker="o",
    ecolor=(0,0,0,0.7),
    capsize=3)
plt.title("Mean Number of Steps Taken to Relax State\nby Number of Units Updated")
plt.xlabel("Number of Units Updated (Per Network Step)")
plt.ylabel("Mean Number of Steps Taken to Relax State")
plt.show()