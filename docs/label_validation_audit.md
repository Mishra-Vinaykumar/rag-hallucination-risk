# Final hallucination-label validation audit

Validation status: **Passed**

The human annotation is the final target. Automated NLI output is used
only to identify cases that require evidence review; it never overrides
the frozen annotation rubric.

- Audited responses: 60
- Abstentions confirmed: 26
- Human/NLI agreements confirmed: 23
- NLI disagreements reviewed: 11
- Duplicate response IDs: 0
- Missing final labels: 0

## Reviewed disagreements

### 5ade025e5542997dc790711e

- Question: In which American football game was Malcolm Smith named Most Valuable   player?
- Human / automatic label: 0 / 1
- Review decision: retain human label 0
- Evidence rationale: Retrieved context directly states that Malcolm Smith was named Most Valuable Player of Super Bowl XLVIII.

### 5ae0d91e55429924de1b7198

- Question: The 1988 American comedy film, The Great Outdoors, starred a four-time   Academy Award nominee, who received a star on the Hollywood Walk of Fame in   what year?
- Human / automatic label: 0 / 1
- Review decision: retain human label 0
- Evidence rationale: Retrieved context lists Annette Bening among the stars of The Great Outdoors and states that she received a Hollywood Walk of Fame star in 2006.

### 5a83168855429966c78a6b2e

- Question: Dua Lipa, an English singer, songwriter and model, the album spawned the   number-one single "New Rules" is a song by English singer Dua Lipa   from her eponymous debut studio album, released in what year?
- Human / automatic label: 0 / 1
- Review decision: retain human label 0
- Evidence rationale: Retrieved context directly states that New Rules is from Dua Lipa's eponymous debut studio album (2017).

### 5adf732a5542993a75d264e9

- Question: Which  American politician did   Donahue replaced
- Human / automatic label: 0 / 1
- Review decision: retain human label 0
- Evidence rationale: Retrieved context directly states that Donahue replaced Kelli Ward after Ward resigned.

### 5ac2a912554299218029dae8

- Question: Which band was founded first, Hole, the rock band that Courtney Love was   a frontwoman of, or The Wolfhounds?
- Human / automatic label: 1 / 0
- Review decision: retain human label 1
- Evidence rationale: Retrieved context states that Hole was formed in 1989 and The Wolfhounds in 1985, so the generated answer Hole is contradicted by the retrieved evidence.

### 5a8b63755542997f31a41cfe

- Question: 750 7th Avenue and 101 Park Avenue, are located in which city?
- Human / automatic label: 0 / 1
- Review decision: retain human label 0
- Evidence rationale: Retrieved context directly states that both 750 Seventh Avenue and 101 Park Avenue are in New York City.

### 5ac15df95542994d76dccded

- Question: Which actress played the part of fictitious character Kimberly Ann Hart,   in the franchise built around a live action superhero television series   taking much of its footage from the Japanese tokusatsu 'Super Sentai'?
- Human / automatic label: 0 / 1
- Review decision: retain human label 0
- Evidence rationale: Retrieved context directly states that Kimberly Ann Hart was played by actress Amy Jo Johnson.

### 5add28c85542992ae4cec4be

- Question: The 337th Flight Test Squadron (337 FLTS) was most recently part of the   46th Test Wing and based at McClellan Air Force Base, a former United States   Air Force base located in the North Highlands area of Sacramento County, in   which US state?
- Human / automatic label: 0 / 1
- Review decision: retain human label 0
- Evidence rationale: Retrieved context directly states that McClellan Air Force Base is in California.

### 5a8ae49155429951533613a3

- Question: In what show did Cynthia Nixon receive the 2004 Primetime Emmy Award for   Outstanding Supporting Actress in a Comedy Series and a Screen Actors Guild   Award for her performance?
- Human / automatic label: 0 / 1
- Review decision: retain human label 0
- Evidence rationale: Retrieved context states that Cynthia Nixon portrayed Miranda Hobbes in Sex and the City and won the 2004 Emmy for that role, with the associated performance also receiving a Screen Actors Guild Award.

### 5ab99943554299131ca42391

- Question: The 1895/96 Football League season was the eighth in Football League   history with Everton, their Goodison Park home, is a football stadium located   in Walton, Liverpool, in which country?
- Human / automatic label: 0 / 1
- Review decision: retain human label 0
- Evidence rationale: Retrieved context directly states that Goodison Park is located in Walton, Liverpool, England.

### 5adbe1e755429947ff173853

- Question: The American Pre-Code comedy film featuring an American actress, dancer,   and singer, widely known for performing in films and RKO's musical films, was   released in what year?
- Human / automatic label: 1 / 0
- Review decision: retain human label 1
- Evidence rationale: Retrieved context identifies Ginger Rogers and separately gives 1932 release dates for candidate RKO films, but it does not explicitly link the described actress to the target film, so the year is not fully entailed.
