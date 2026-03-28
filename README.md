# DS 4320 Project 1: Predicting the Men's March Madness Tournament
An analysis of which NCAA Men's Basketball teams are the most successful in the championship tournament, which teams might be most likely to reach the final four and win the title in future rounds, and what teams would come out on top if they were to face each other in the tournament. The model is trained using data from various seasons and includes averages of 

Name: Teagan Britten

Computing ID: uup3cy

DOI: 

### [Access the Press Release Here](pressrelease.md)
### [Access the Data Here](https://1drv.ms/f/c/6a550bae65b9bbfe/IgCLSUE3acp8SqdHENqi9fJtAUzyQvq90YjV4GNCzZ49RTw?e=XurULA)
### [Access the Data Pipeline Here](pipeline.ipynb)

## Problem Definition

### General Problem: 
Predicting sports game outcomes

### Specific Problem: 
Can we use past team and seed results, rankings, roster quality, and other performance metrics to accurately predict the winner of each game of the Men's NCAA Basketball Tournament? 

### Rationale
Sports games, by nature but also with time, involve a significant amount of data that can be analysed to understand how a team performed and why one team outperformed the other. Combining these results into season-long analysis and developing sport-specific metrics for understanding which teams perform better than others provides an even better and more detailed picture of a team's success and what contributes to that. College basketball is certainly no exception to having a vast array of data available to understand it, and has a vast number of competing teams to understand and predict performance for. The NCAA tournament, or March Madness, is the ending of the season, and is one of the most unique and logistically complex sporting events in the world. It involves a variety of teams facing off against others that rarely faced each other and may not even have any common opponents that season. This makes predicting the outcomes of games in the college basketball season and specifically March Madness a difficult yet possible and valuable task to take on. 

### Motivation
Being able to accurately predict the outcome of basketball games could be very valuable to a number of parties. One of these are teams looking at their potential opponents. If a team is preparing for the NCAA tournament, in which turnaround periods between games are incredibly slim, then knowing that one team is far more likely than the other to be the next opponent could help that team decide how to weight their preparations going into that round of the tournament. Athletic departments would also be able to better plan and allocate resources if they have a good understanding of if and when a team might be playing. The organisers of the tournament can also better prepare for what accommodations are needed for specific teams, prepare to sell tickets for said teams, and advertise to fans of those teams that might want to attend their games.

## Domain Exposition

### Table 1: Definitions

| Term | Definition |
| --- | --- |
| NET (NCAA Evaluation Tool) | An algorithmic metric used to rank all division I teams based on their performance |
| FG% | Field Goal Percentage is the proportion of made shots from the field of all attempts |
| eFG% | effective Field Goal percentage is a metric using field goal percentage to 
| Seed | Where the team is ranked 1-16 within one of four regions of the tournament |
| Bubble | The ranking area for teams that are on the edge of making the tournament, either in the 'field' or out |
| WAB | Wins Above Bubble represents the difference in the number of wins a team has compared to the expected number of wins an average "bubble" team would earn against a given teams' schedule |
| Quadrants | Games for teams are divided into four quadrants based on opponent quality and location, with Quadrant 1 games being the most difficult and Quadrant 4 wins being the least difficult |

### Paragraph:
NCAA Division I Men's Basketball features over 300 teams in 31 leagues known as conferences. Each year, the season finishes with a 68-team tournament of which the winner is declared the national champion. The winner of each league is given automatic entry into the tournament, with the remaining teams selected by a committee based on their performance across the season, referred to as their resume. The bubble includes the lowest ranked at-large teams (teams without an automatic entry) that have reasonable potential to qualify for the tournament. A bid refers to a team being invited to participate in the tournament. 

### Table 2: Domain Articles

[Link to All Articles](https://1drv.ms/f/c/6a550bae65b9bbfe/IgDpPy3oLBabSIEaHf0wTiNBAYl-XSNj_kqXjkrSQjjIEok?e=Kg92FL)

| Article Name | Summary | Link |
| --- | --- | --- |
| Probabilities of Victory in Head-to-Head Team Matchups | A detailed explanation of predicting the outcome of sports games, applied to the sport of baseball | https://1drv.ms/b/c/6a550bae65b9bbfe/IQCoo7Y-FsfkSbSU9fKfzrn5AUnbsLPBb3uv1CKfmNsQ8RE?e=n3E2e6 |
| March Madness bracketology: The ultimate guide | Provides an overview of how the tournament is set-up and how the bracket is formed | https://www.ncaa.com/news/basketball-men/article/2020-01-14/march-madness-bracketology-ultimate-guide |
| The Ultimate Guide to Predictive College Basketball Analytics | Covers a wide variety of predictive analytics and how they are calculated | https://thepowerrank.com/cbb-analytics/ |
| Analytics in College Hoops: A New Type of March Madness | This article, using Michigan State as an example, covers how teams use data-driven processes to inform their decision making | https://www.michiganstateuniversityonline.com/resources/business-analytics/analytics-in-college-hoops-a-new-type-of-march-madness/ |
| EvanMiya.com Resume Metrics | A database of various calculated metrics for ranking teams in the 25/26 season | https://evanmiya.com/?resume_metrics |
| The science of strength: How data analytics is transforming college basketball | MIT looks at how data impacts basketball teams beyond just predictive metrics | https://1drv.ms/b/c/6a550bae65b9bbfe/IQBoxziryfllQ6zidZEQHIlUAalGdfUrrLYczeK6vWCv-sY?e=FQ21A7 |

