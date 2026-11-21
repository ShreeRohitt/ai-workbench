# Learnings

Written by me, in my own words, after each milestone. (Claude does not write this file.)

29/10/2026 <milestone 1>
- What I expected: That I'll do metric wise impact on data through visualisation and find which metrics are useful and remove not important ones
- What actually happened (with numbers): I found correlations, I found timing of records mattter, metrics like gender shouldn't be taken into account.
- What I'd explain differently in an interview now: I'll clearly explain how I did EDA, with numbers, make them understand ceiling and uncertainity(+-18).

05/11/2026 <milestone 2>
- I expected logistic regression to perform better than the rule based 
- LR actually performed well with a PR of 0.6 at K10% beating the rule(0.48) and random(0.14) and we also looked at LR with survey column which did well as expected beating LR 22/25 times but we can't proceed with it further because it would be causing leakage as we don't know the timings of the survey. Then we tried a combination of LR with the rule column which showed a little improvement but it was not worth the effort.
- I'd explain LR's precision comparing it with the preceders, I'll explain why we are not taking in the urvey columns and how LR with the rule column isn't worth the effort as it just shows +0.8 PR AUC improvement, and we don't know if it was noise.

21/11/2026 <milestone 3>
- What I expected: I expected the model would perform very well than classic LR because it'll learn interactions and find combinations of columns.
- What actually happened (with numbers): to my surprise it didn't get any closer to my bet of plus 0.20, it was just 0.020 improvement in PR-AUC (18 wins / 7 losses on the same 25 folds), and it doesn't have to do anything with the model. It's the limitation of the data: we'd need more informative columns, not just more rows (LR's fit-vs-eval gap is only +0.056). We tried nested cv tuning (+0.002 vs untuned LightGBM) and introducing class weights (−0.024 for LR, which hurt on 23 of 25 folds, and −0.007 for LightGBM, which was noise), so neither helped. We concluded with capacity at k=30 (per fold of ~235 employees): we could potentially identify 17 leavers of the 38, with a precision of 0.565 ± 0.083, whereas random would only get us 4.8. And we're proceeding with LR as it is simple and does the job. I also found out that when we increase the number of calls we aren't doing much better: calls 51–60 find 1.5 more leavers, no better than random (1.6), because the model puts leavers at the top. The ± shrinks as the denominator increases, so one leaver doesn't cause a big impact at higher capacity.
- What I'd explain differently in an interview now: (to write after the 03-M3 interview)
