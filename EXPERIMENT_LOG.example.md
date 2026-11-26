# Experiment log
HUMANS ONLY (the launcher appends job links after tags). The launcher refuses any ablation whose backticked tag is not
on a `- ` line under its `## <exp>` heading, or whose section lacks QUESTION and HYPOTHESIS. Format:

```
## <exp> (branch: <exp>)
QUESTION: <question>
HYPOTHESIS: <hypothesis>

- YYYY-MM-DD HH:MM EST `<ablation>`: <goal to be established, what will cause that, and what we are testing>
<!-- NOTES
Changes:
Expected:
Results:
-->
    - YYYY-MM-DD HH:MM EST `<sub-ablation or setup>`: <one-liner, for cluster runs that are subablations, proof-of-concepts, smokes, or setup for the actual main hypothesis>
```

Failed experiments: entry in main's log with branch, code changes, why it failed.
