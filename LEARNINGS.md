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
