
1. The model will be used the HR team to go people who are likely to leave and have a conversation to retain them.
2. k is the capacity of hr team, the number of people they can call. Example if there are 2 HRs, they can call 15*2, 30 people. k is set by HR's capacity, not by the model. Here it's an assumption, since there's no real HR team
3. Out of the top k list, how many actually left is precision. With 16% baseline, if the k is 30, 16 % of 30 is 5, so the model should beat this. The model should give more right employees in the top k.
4. A false negative hurts more becuase losing employee hurts, hence we should make the best out of it. 