# Tiao and Box test data

`gas_furnace.csv` — Box and Jenkins' Series J (1970, *Time Series Analysis:
Forecasting and Control*): input gas rate and output CO2 (%), 296 pairs at
9-second intervals. Downloaded 2026-10-02 from <https://openmv.net/info/gas-furnace>
(md5 `4950b8ad0b048c453257747ea6e04078`). The input's mean is −0.0568, which matches
Tiao and Box's (1981, §5.2) centring Z₁ = gas rate + .057.

Tiao and Box (1981) Table 12(b) publishes the stepwise M(l) on these data;
`tests/test_tiao_box.py` reproduces it.

Box and Tiao's (1977) hog-data moments (μ̂, C₀, C₁, p. 359) are typed in the
test itself: the raw series are not printed in the paper.
