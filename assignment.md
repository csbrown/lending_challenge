

```python
import pandas as pd
```


```python
import numpy as np
```


```python
import collections
```


```python
import sklearn, sklearn.tree
```


```python
df = pd.read_csv("data.csv")
```


```python
df["days_delinquent_old_bin"] = pd.cut(df["days_delinquent_old"], bins=[0,1,5,10,30,60,np.inf], include_lowest=True, right=False)
```


```python
df["days_delinquent_new_bin"] = pd.cut(df["days_delinquent_new"], bins=[0,1,5,10,30,60,np.inf], include_lowest=True, right=False)
```


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>as_of_date</th>
      <th>days_delinquent_old</th>
      <th>days_delinquent_new</th>
      <th>new_outstanding_principal_balance</th>
      <th>initial_loan_amount</th>
      <th>fico</th>
      <th>sales_channel__c</th>
      <th>type</th>
      <th>current_collection_method</th>
      <th>term</th>
      <th>lender_payoff</th>
      <th>average_bank_balance__c</th>
      <th>last_cleared_payment_date</th>
      <th>days_delinquent_old_bin</th>
      <th>days_delinquent_new_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-11-01</td>
      <td>180</td>
      <td>180</td>
      <td>29383.75</td>
      <td>50000</td>
      <td>641.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>0.00</td>
      <td>46445.21</td>
      <td>2012-11-01</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-11-01</td>
      <td>9</td>
      <td>9</td>
      <td>3200.41</td>
      <td>10000</td>
      <td>631.0</td>
      <td>Referral</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>1284.07</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[5, 10)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-11-01</td>
      <td>56</td>
      <td>75</td>
      <td>56207.38</td>
      <td>60000</td>
      <td>671.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>9</td>
      <td>0.00</td>
      <td>29415.70</td>
      <td>2012-07-30</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-11-01</td>
      <td>19</td>
      <td>30</td>
      <td>47496.37</td>
      <td>50000</td>
      <td>626.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>61028.12</td>
      <td>2012-10-04</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-11-01</td>
      <td>35</td>
      <td>54</td>
      <td>21012.15</td>
      <td>25000</td>
      <td>587.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2046.37</td>
      <td>2012-09-07</td>
      <td>[30, 60)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2012-11-01</td>
      <td>12</td>
      <td>26</td>
      <td>15423.23</td>
      <td>20000</td>
      <td>706.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>0.00</td>
      <td>2773.77</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2012-11-01</td>
      <td>180</td>
      <td>180</td>
      <td>1409.18</td>
      <td>25000</td>
      <td>NaN</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>10393.84</td>
      <td>2012-09-17</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>6</td>
      <td>8556.03</td>
      <td>10000</td>
      <td>654.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2346.46</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[5, 10)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>0</td>
      <td>809.36</td>
      <td>10000</td>
      <td>593.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2262.00</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[0, 1)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2012-11-01</td>
      <td>46</td>
      <td>65</td>
      <td>2381.85</td>
      <td>15000</td>
      <td>537.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>1407.21</td>
      <td>3176.13</td>
      <td>2012-08-27</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2012-11-01</td>
      <td>3</td>
      <td>6</td>
      <td>23345.52</td>
      <td>25000</td>
      <td>716.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>6</td>
      <td>0.00</td>
      <td>423.57</td>
      <td>2012-11-01</td>
      <td>[1, 5)</td>
      <td>[5, 10)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2012-11-01</td>
      <td>2</td>
      <td>0</td>
      <td>20395.49</td>
      <td>30000</td>
      <td>631.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2105.21</td>
      <td>2012-11-01</td>
      <td>[1, 5)</td>
      <td>[0, 1)</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>0</td>
      <td>17000.00</td>
      <td>17000</td>
      <td>647.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3905.75</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[0, 1)</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2012-11-01</td>
      <td>47</td>
      <td>66</td>
      <td>18719.52</td>
      <td>20000</td>
      <td>541.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>5756.67</td>
      <td>2012-08-28</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2012-11-01</td>
      <td>4</td>
      <td>9</td>
      <td>15000.00</td>
      <td>15000</td>
      <td>734.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2994.42</td>
      <td>2012-10-30</td>
      <td>[1, 5)</td>
      <td>[5, 10)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2012-11-01</td>
      <td>3</td>
      <td>13</td>
      <td>58902.55</td>
      <td>75000</td>
      <td>686.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>0.00</td>
      <td>15785.22</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2012-11-01</td>
      <td>61</td>
      <td>80</td>
      <td>20000.00</td>
      <td>20000</td>
      <td>687.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3291.55</td>
      <td>2012-07-11</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2012-11-01</td>
      <td>29</td>
      <td>42</td>
      <td>35000.00</td>
      <td>35000</td>
      <td>584.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>6</td>
      <td>8689.75</td>
      <td>2072.33</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2012-11-01</td>
      <td>61</td>
      <td>68</td>
      <td>6158.95</td>
      <td>17500</td>
      <td>611.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3603.82</td>
      <td>2012-11-01</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2012-11-01</td>
      <td>6</td>
      <td>1</td>
      <td>3888.58</td>
      <td>10000</td>
      <td>694.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2833.90</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[1, 5)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2012-11-01</td>
      <td>2</td>
      <td>2</td>
      <td>61915.93</td>
      <td>75000</td>
      <td>648.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>37500.00</td>
      <td>7804.34</td>
      <td>2012-11-01</td>
      <td>[1, 5)</td>
      <td>[1, 5)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2012-11-01</td>
      <td>180</td>
      <td>180</td>
      <td>15817.58</td>
      <td>50000</td>
      <td>794.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>9</td>
      <td>0.00</td>
      <td>15080.60</td>
      <td>2012-10-19</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2012-11-01</td>
      <td>17</td>
      <td>33</td>
      <td>33000.00</td>
      <td>33000</td>
      <td>619.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>18</td>
      <td>16892.70</td>
      <td>4245.33</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2012-11-01</td>
      <td>41</td>
      <td>60</td>
      <td>35000.00</td>
      <td>35000</td>
      <td>689.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>49921.18</td>
      <td>2012-09-04</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2012-11-01</td>
      <td>21</td>
      <td>38</td>
      <td>13585.53</td>
      <td>35000</td>
      <td>696.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>34422.00</td>
      <td>2012-10-26</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2012-11-01</td>
      <td>4</td>
      <td>23</td>
      <td>8150.82</td>
      <td>10000</td>
      <td>620.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2594.94</td>
      <td>2012-10-26</td>
      <td>[1, 5)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2012-11-01</td>
      <td>27</td>
      <td>0</td>
      <td>807.42</td>
      <td>12000</td>
      <td>589.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>1380.65</td>
      <td>2012-10-18</td>
      <td>[10, 30)</td>
      <td>[0, 1)</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2012-11-01</td>
      <td>6</td>
      <td>0</td>
      <td>6101.18</td>
      <td>10000</td>
      <td>710.0</td>
      <td>Referral</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>1132.31</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[0, 1)</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>0</td>
      <td>91478.80</td>
      <td>100000</td>
      <td>735.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>33993.01</td>
      <td>2012-11-01</td>
      <td>[1, 5)</td>
      <td>[0, 1)</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2012-11-01</td>
      <td>4</td>
      <td>12</td>
      <td>20000.00</td>
      <td>20000</td>
      <td>624.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>12</td>
      <td>0.00</td>
      <td>1425.57</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>447</th>
      <td>2012-11-01</td>
      <td>57</td>
      <td>76</td>
      <td>6453.42</td>
      <td>10000</td>
      <td>654.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3493.52</td>
      <td>2012-09-13</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>448</th>
      <td>2012-11-01</td>
      <td>56</td>
      <td>75</td>
      <td>25612.32</td>
      <td>35000</td>
      <td>605.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>8333.71</td>
      <td>2012-08-13</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>449</th>
      <td>2012-11-01</td>
      <td>9</td>
      <td>28</td>
      <td>35000.00</td>
      <td>35000</td>
      <td>683.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>12</td>
      <td>0.00</td>
      <td>3878.88</td>
      <td>2012-10-31</td>
      <td>[5, 10)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>450</th>
      <td>2012-11-01</td>
      <td>7</td>
      <td>12</td>
      <td>26316.51</td>
      <td>35000</td>
      <td>509.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>17236.00</td>
      <td>4931.75</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>451</th>
      <td>2012-11-01</td>
      <td>11</td>
      <td>19</td>
      <td>3721.46</td>
      <td>12000</td>
      <td>595.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>6327.60</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>452</th>
      <td>2012-11-01</td>
      <td>180</td>
      <td>180</td>
      <td>22510.35</td>
      <td>50000</td>
      <td>641.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>16840.99</td>
      <td>46445.21</td>
      <td>2012-11-01</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>453</th>
      <td>2012-11-01</td>
      <td>62</td>
      <td>64</td>
      <td>7230.36</td>
      <td>20000</td>
      <td>536.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>11730.96</td>
      <td>6213.26</td>
      <td>2012-11-01</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>454</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>3</td>
      <td>10041.28</td>
      <td>15000</td>
      <td>625.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2159.63</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[1, 5)</td>
    </tr>
    <tr>
      <th>455</th>
      <td>2012-11-01</td>
      <td>27</td>
      <td>37</td>
      <td>17960.27</td>
      <td>23000</td>
      <td>596.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>6</td>
      <td>12520.00</td>
      <td>3209.72</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>456</th>
      <td>2012-11-01</td>
      <td>2</td>
      <td>8</td>
      <td>13978.52</td>
      <td>18000</td>
      <td>625.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>4447.46</td>
      <td>2012-10-30</td>
      <td>[1, 5)</td>
      <td>[5, 10)</td>
    </tr>
    <tr>
      <th>457</th>
      <td>2012-11-01</td>
      <td>29</td>
      <td>48</td>
      <td>7006.53</td>
      <td>10000</td>
      <td>683.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3022.38</td>
      <td>2012-09-20</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>458</th>
      <td>2012-11-01</td>
      <td>14</td>
      <td>0</td>
      <td>714.43</td>
      <td>35000</td>
      <td>637.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>5632.82</td>
      <td>2012-10-31</td>
      <td>[10, 30)</td>
      <td>[0, 1)</td>
    </tr>
    <tr>
      <th>459</th>
      <td>2012-11-01</td>
      <td>5</td>
      <td>15</td>
      <td>18000.00</td>
      <td>18000</td>
      <td>606.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>6</td>
      <td>0.00</td>
      <td>2941.75</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>460</th>
      <td>2012-11-01</td>
      <td>42</td>
      <td>42</td>
      <td>19847.95</td>
      <td>35000</td>
      <td>592.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>10130.56</td>
      <td>2012-11-01</td>
      <td>[30, 60)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>461</th>
      <td>2012-11-01</td>
      <td>45</td>
      <td>64</td>
      <td>19577.45</td>
      <td>35000</td>
      <td>657.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>9</td>
      <td>0.00</td>
      <td>9448.96</td>
      <td>2012-09-06</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>462</th>
      <td>2012-11-01</td>
      <td>17</td>
      <td>16</td>
      <td>10911.14</td>
      <td>25000</td>
      <td>536.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>7024.00</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>463</th>
      <td>2012-11-01</td>
      <td>18</td>
      <td>0</td>
      <td>3067.39</td>
      <td>20000</td>
      <td>667.0</td>
      <td>Direct</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2758.55</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[0, 1)</td>
    </tr>
    <tr>
      <th>464</th>
      <td>2012-11-01</td>
      <td>17</td>
      <td>36</td>
      <td>32868.71</td>
      <td>35000</td>
      <td>658.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>15800.50</td>
      <td>2012-10-17</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>465</th>
      <td>2012-11-01</td>
      <td>7</td>
      <td>7</td>
      <td>11170.08</td>
      <td>30000</td>
      <td>692.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>9</td>
      <td>0.00</td>
      <td>13078.80</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[5, 10)</td>
    </tr>
    <tr>
      <th>466</th>
      <td>2012-11-01</td>
      <td>13</td>
      <td>13</td>
      <td>41509.35</td>
      <td>50000</td>
      <td>695.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>0.00</td>
      <td>17441.15</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>467</th>
      <td>2012-11-01</td>
      <td>61</td>
      <td>80</td>
      <td>40000.00</td>
      <td>40000</td>
      <td>689.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>13461.46</td>
      <td>2012-07-26</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>468</th>
      <td>2012-11-01</td>
      <td>17</td>
      <td>36</td>
      <td>7771.34</td>
      <td>8000</td>
      <td>532.0</td>
      <td>Referral</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3526.42</td>
      <td>2012-10-19</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>469</th>
      <td>2012-11-01</td>
      <td>8</td>
      <td>25</td>
      <td>7287.10</td>
      <td>10000</td>
      <td>697.0</td>
      <td>Direct</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>1785.80</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>470</th>
      <td>2012-11-01</td>
      <td>4</td>
      <td>21</td>
      <td>4414.44</td>
      <td>15000</td>
      <td>717.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>15127.40</td>
      <td>2012-10-26</td>
      <td>[1, 5)</td>
      <td>[10, 30)</td>
    </tr>
    <tr>
      <th>471</th>
      <td>2012-11-01</td>
      <td>58</td>
      <td>77</td>
      <td>6890.41</td>
      <td>40000</td>
      <td>613.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>21526.16</td>
      <td>2012-08-13</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>472</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>0</td>
      <td>93541.75</td>
      <td>100000</td>
      <td>645.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>9</td>
      <td>0.00</td>
      <td>13328.08</td>
      <td>2012-11-01</td>
      <td>[1, 5)</td>
      <td>[0, 1)</td>
    </tr>
    <tr>
      <th>473</th>
      <td>2012-11-01</td>
      <td>62</td>
      <td>81</td>
      <td>14697.31</td>
      <td>40000</td>
      <td>583.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>38779.38</td>
      <td>2012-07-23</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
    </tr>
    <tr>
      <th>474</th>
      <td>2012-11-01</td>
      <td>39</td>
      <td>58</td>
      <td>15000.00</td>
      <td>15000</td>
      <td>518.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>5</td>
      <td>0.00</td>
      <td>14469.02</td>
      <td>2012-09-12</td>
      <td>[30, 60)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>475</th>
      <td>2012-11-01</td>
      <td>19</td>
      <td>38</td>
      <td>9355.63</td>
      <td>10000</td>
      <td>578.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2463.21</td>
      <td>2012-10-04</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
    </tr>
    <tr>
      <th>476</th>
      <td>2012-11-01</td>
      <td>22</td>
      <td>41</td>
      <td>75000.00</td>
      <td>75000</td>
      <td>625.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>12</td>
      <td>0.00</td>
      <td>14640.79</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
    </tr>
  </tbody>
</table>
<p>477 rows × 15 columns</p>
</div>




```python
def transition_matrix(series1, series2, weights=None):
    transition_matrix = pd.crosstab(series1, series2, values=weights, aggfunc=sum if weights is not None else None, normalize='index')
    return transition_matrix
    
def test_transition_matrix():
    d = pd.DataFrame([[1,2,3],[3,4,1],[1,4,1]])
    tm1 = transition_matrix(d[0],d[1])
    tm2 = transition_matrix(d[0],d[1],d[2])
    assert np.allclose(tm1.as_matrix(), np.array([[0.5,0.5],[0.0,1.0]]), rtol=0.001, atol=0)
    assert np.allclose(tm2.as_matrix(), np.array([[0.75,0.25],[0.0,1.0]]), rtol=0.001, atol=0)
```


```python
test_transition_matrix()
```


```python
df["gain"] = df["days_delinquent_new"] - df["days_delinquent_old"]
```


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>as_of_date</th>
      <th>days_delinquent_old</th>
      <th>days_delinquent_new</th>
      <th>new_outstanding_principal_balance</th>
      <th>initial_loan_amount</th>
      <th>fico</th>
      <th>sales_channel__c</th>
      <th>type</th>
      <th>current_collection_method</th>
      <th>term</th>
      <th>lender_payoff</th>
      <th>average_bank_balance__c</th>
      <th>last_cleared_payment_date</th>
      <th>days_delinquent_old_bin</th>
      <th>days_delinquent_new_bin</th>
      <th>gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2012-11-01</td>
      <td>180</td>
      <td>180</td>
      <td>29383.75</td>
      <td>50000</td>
      <td>641.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>0.00</td>
      <td>46445.21</td>
      <td>2012-11-01</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2012-11-01</td>
      <td>9</td>
      <td>9</td>
      <td>3200.41</td>
      <td>10000</td>
      <td>631.0</td>
      <td>Referral</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>1284.07</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[5, 10)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-11-01</td>
      <td>56</td>
      <td>75</td>
      <td>56207.38</td>
      <td>60000</td>
      <td>671.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>9</td>
      <td>0.00</td>
      <td>29415.70</td>
      <td>2012-07-30</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2012-11-01</td>
      <td>19</td>
      <td>30</td>
      <td>47496.37</td>
      <td>50000</td>
      <td>626.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>61028.12</td>
      <td>2012-10-04</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2012-11-01</td>
      <td>35</td>
      <td>54</td>
      <td>21012.15</td>
      <td>25000</td>
      <td>587.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2046.37</td>
      <td>2012-09-07</td>
      <td>[30, 60)</td>
      <td>[30, 60)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2012-11-01</td>
      <td>12</td>
      <td>26</td>
      <td>15423.23</td>
      <td>20000</td>
      <td>706.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>0.00</td>
      <td>2773.77</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[10, 30)</td>
      <td>14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2012-11-01</td>
      <td>180</td>
      <td>180</td>
      <td>1409.18</td>
      <td>25000</td>
      <td>NaN</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>10393.84</td>
      <td>2012-09-17</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>6</td>
      <td>8556.03</td>
      <td>10000</td>
      <td>654.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2346.46</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[5, 10)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>0</td>
      <td>809.36</td>
      <td>10000</td>
      <td>593.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2262.00</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[0, 1)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2012-11-01</td>
      <td>46</td>
      <td>65</td>
      <td>2381.85</td>
      <td>15000</td>
      <td>537.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>1407.21</td>
      <td>3176.13</td>
      <td>2012-08-27</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2012-11-01</td>
      <td>3</td>
      <td>6</td>
      <td>23345.52</td>
      <td>25000</td>
      <td>716.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>6</td>
      <td>0.00</td>
      <td>423.57</td>
      <td>2012-11-01</td>
      <td>[1, 5)</td>
      <td>[5, 10)</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2012-11-01</td>
      <td>2</td>
      <td>0</td>
      <td>20395.49</td>
      <td>30000</td>
      <td>631.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2105.21</td>
      <td>2012-11-01</td>
      <td>[1, 5)</td>
      <td>[0, 1)</td>
      <td>-2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>0</td>
      <td>17000.00</td>
      <td>17000</td>
      <td>647.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3905.75</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[0, 1)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2012-11-01</td>
      <td>47</td>
      <td>66</td>
      <td>18719.52</td>
      <td>20000</td>
      <td>541.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>5756.67</td>
      <td>2012-08-28</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2012-11-01</td>
      <td>4</td>
      <td>9</td>
      <td>15000.00</td>
      <td>15000</td>
      <td>734.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2994.42</td>
      <td>2012-10-30</td>
      <td>[1, 5)</td>
      <td>[5, 10)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2012-11-01</td>
      <td>3</td>
      <td>13</td>
      <td>58902.55</td>
      <td>75000</td>
      <td>686.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>0.00</td>
      <td>15785.22</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[10, 30)</td>
      <td>10</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2012-11-01</td>
      <td>61</td>
      <td>80</td>
      <td>20000.00</td>
      <td>20000</td>
      <td>687.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3291.55</td>
      <td>2012-07-11</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2012-11-01</td>
      <td>29</td>
      <td>42</td>
      <td>35000.00</td>
      <td>35000</td>
      <td>584.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>6</td>
      <td>8689.75</td>
      <td>2072.33</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
      <td>13</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2012-11-01</td>
      <td>61</td>
      <td>68</td>
      <td>6158.95</td>
      <td>17500</td>
      <td>611.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3603.82</td>
      <td>2012-11-01</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
      <td>7</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2012-11-01</td>
      <td>6</td>
      <td>1</td>
      <td>3888.58</td>
      <td>10000</td>
      <td>694.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2833.90</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[1, 5)</td>
      <td>-5</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2012-11-01</td>
      <td>2</td>
      <td>2</td>
      <td>61915.93</td>
      <td>75000</td>
      <td>648.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>37500.00</td>
      <td>7804.34</td>
      <td>2012-11-01</td>
      <td>[1, 5)</td>
      <td>[1, 5)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2012-11-01</td>
      <td>180</td>
      <td>180</td>
      <td>15817.58</td>
      <td>50000</td>
      <td>794.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>9</td>
      <td>0.00</td>
      <td>15080.60</td>
      <td>2012-10-19</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2012-11-01</td>
      <td>17</td>
      <td>33</td>
      <td>33000.00</td>
      <td>33000</td>
      <td>619.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>18</td>
      <td>16892.70</td>
      <td>4245.33</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
      <td>16</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2012-11-01</td>
      <td>41</td>
      <td>60</td>
      <td>35000.00</td>
      <td>35000</td>
      <td>689.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>49921.18</td>
      <td>2012-09-04</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2012-11-01</td>
      <td>21</td>
      <td>38</td>
      <td>13585.53</td>
      <td>35000</td>
      <td>696.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>34422.00</td>
      <td>2012-10-26</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
      <td>17</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2012-11-01</td>
      <td>4</td>
      <td>23</td>
      <td>8150.82</td>
      <td>10000</td>
      <td>620.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2594.94</td>
      <td>2012-10-26</td>
      <td>[1, 5)</td>
      <td>[10, 30)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2012-11-01</td>
      <td>27</td>
      <td>0</td>
      <td>807.42</td>
      <td>12000</td>
      <td>589.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>1380.65</td>
      <td>2012-10-18</td>
      <td>[10, 30)</td>
      <td>[0, 1)</td>
      <td>-27</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2012-11-01</td>
      <td>6</td>
      <td>0</td>
      <td>6101.18</td>
      <td>10000</td>
      <td>710.0</td>
      <td>Referral</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>1132.31</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[0, 1)</td>
      <td>-6</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>0</td>
      <td>91478.80</td>
      <td>100000</td>
      <td>735.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>33993.01</td>
      <td>2012-11-01</td>
      <td>[1, 5)</td>
      <td>[0, 1)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2012-11-01</td>
      <td>4</td>
      <td>12</td>
      <td>20000.00</td>
      <td>20000</td>
      <td>624.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>12</td>
      <td>0.00</td>
      <td>1425.57</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[10, 30)</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>447</th>
      <td>2012-11-01</td>
      <td>57</td>
      <td>76</td>
      <td>6453.42</td>
      <td>10000</td>
      <td>654.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3493.52</td>
      <td>2012-09-13</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>448</th>
      <td>2012-11-01</td>
      <td>56</td>
      <td>75</td>
      <td>25612.32</td>
      <td>35000</td>
      <td>605.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>8333.71</td>
      <td>2012-08-13</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>449</th>
      <td>2012-11-01</td>
      <td>9</td>
      <td>28</td>
      <td>35000.00</td>
      <td>35000</td>
      <td>683.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>12</td>
      <td>0.00</td>
      <td>3878.88</td>
      <td>2012-10-31</td>
      <td>[5, 10)</td>
      <td>[10, 30)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>450</th>
      <td>2012-11-01</td>
      <td>7</td>
      <td>12</td>
      <td>26316.51</td>
      <td>35000</td>
      <td>509.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>17236.00</td>
      <td>4931.75</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[10, 30)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>451</th>
      <td>2012-11-01</td>
      <td>11</td>
      <td>19</td>
      <td>3721.46</td>
      <td>12000</td>
      <td>595.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>6327.60</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[10, 30)</td>
      <td>8</td>
    </tr>
    <tr>
      <th>452</th>
      <td>2012-11-01</td>
      <td>180</td>
      <td>180</td>
      <td>22510.35</td>
      <td>50000</td>
      <td>641.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>16840.99</td>
      <td>46445.21</td>
      <td>2012-11-01</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>453</th>
      <td>2012-11-01</td>
      <td>62</td>
      <td>64</td>
      <td>7230.36</td>
      <td>20000</td>
      <td>536.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>11730.96</td>
      <td>6213.26</td>
      <td>2012-11-01</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>454</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>3</td>
      <td>10041.28</td>
      <td>15000</td>
      <td>625.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2159.63</td>
      <td>2012-10-31</td>
      <td>[1, 5)</td>
      <td>[1, 5)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>455</th>
      <td>2012-11-01</td>
      <td>27</td>
      <td>37</td>
      <td>17960.27</td>
      <td>23000</td>
      <td>596.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>6</td>
      <td>12520.00</td>
      <td>3209.72</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
      <td>10</td>
    </tr>
    <tr>
      <th>456</th>
      <td>2012-11-01</td>
      <td>2</td>
      <td>8</td>
      <td>13978.52</td>
      <td>18000</td>
      <td>625.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>4447.46</td>
      <td>2012-10-30</td>
      <td>[1, 5)</td>
      <td>[5, 10)</td>
      <td>6</td>
    </tr>
    <tr>
      <th>457</th>
      <td>2012-11-01</td>
      <td>29</td>
      <td>48</td>
      <td>7006.53</td>
      <td>10000</td>
      <td>683.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3022.38</td>
      <td>2012-09-20</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>458</th>
      <td>2012-11-01</td>
      <td>14</td>
      <td>0</td>
      <td>714.43</td>
      <td>35000</td>
      <td>637.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>5632.82</td>
      <td>2012-10-31</td>
      <td>[10, 30)</td>
      <td>[0, 1)</td>
      <td>-14</td>
    </tr>
    <tr>
      <th>459</th>
      <td>2012-11-01</td>
      <td>5</td>
      <td>15</td>
      <td>18000.00</td>
      <td>18000</td>
      <td>606.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>6</td>
      <td>0.00</td>
      <td>2941.75</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[10, 30)</td>
      <td>10</td>
    </tr>
    <tr>
      <th>460</th>
      <td>2012-11-01</td>
      <td>42</td>
      <td>42</td>
      <td>19847.95</td>
      <td>35000</td>
      <td>592.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>10130.56</td>
      <td>2012-11-01</td>
      <td>[30, 60)</td>
      <td>[30, 60)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>461</th>
      <td>2012-11-01</td>
      <td>45</td>
      <td>64</td>
      <td>19577.45</td>
      <td>35000</td>
      <td>657.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>9</td>
      <td>0.00</td>
      <td>9448.96</td>
      <td>2012-09-06</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>462</th>
      <td>2012-11-01</td>
      <td>17</td>
      <td>16</td>
      <td>10911.14</td>
      <td>25000</td>
      <td>536.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>7024.00</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[10, 30)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>463</th>
      <td>2012-11-01</td>
      <td>18</td>
      <td>0</td>
      <td>3067.39</td>
      <td>20000</td>
      <td>667.0</td>
      <td>Direct</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2758.55</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[0, 1)</td>
      <td>-18</td>
    </tr>
    <tr>
      <th>464</th>
      <td>2012-11-01</td>
      <td>17</td>
      <td>36</td>
      <td>32868.71</td>
      <td>35000</td>
      <td>658.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>15800.50</td>
      <td>2012-10-17</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>465</th>
      <td>2012-11-01</td>
      <td>7</td>
      <td>7</td>
      <td>11170.08</td>
      <td>30000</td>
      <td>692.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>9</td>
      <td>0.00</td>
      <td>13078.80</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[5, 10)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>466</th>
      <td>2012-11-01</td>
      <td>13</td>
      <td>13</td>
      <td>41509.35</td>
      <td>50000</td>
      <td>695.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>12</td>
      <td>0.00</td>
      <td>17441.15</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[10, 30)</td>
      <td>0</td>
    </tr>
    <tr>
      <th>467</th>
      <td>2012-11-01</td>
      <td>61</td>
      <td>80</td>
      <td>40000.00</td>
      <td>40000</td>
      <td>689.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>13461.46</td>
      <td>2012-07-26</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>468</th>
      <td>2012-11-01</td>
      <td>17</td>
      <td>36</td>
      <td>7771.34</td>
      <td>8000</td>
      <td>532.0</td>
      <td>Referral</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>3526.42</td>
      <td>2012-10-19</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>469</th>
      <td>2012-11-01</td>
      <td>8</td>
      <td>25</td>
      <td>7287.10</td>
      <td>10000</td>
      <td>697.0</td>
      <td>Direct</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>1785.80</td>
      <td>2012-11-01</td>
      <td>[5, 10)</td>
      <td>[10, 30)</td>
      <td>17</td>
    </tr>
    <tr>
      <th>470</th>
      <td>2012-11-01</td>
      <td>4</td>
      <td>21</td>
      <td>4414.44</td>
      <td>15000</td>
      <td>717.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>15127.40</td>
      <td>2012-10-26</td>
      <td>[1, 5)</td>
      <td>[10, 30)</td>
      <td>17</td>
    </tr>
    <tr>
      <th>471</th>
      <td>2012-11-01</td>
      <td>58</td>
      <td>77</td>
      <td>6890.41</td>
      <td>40000</td>
      <td>613.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>21526.16</td>
      <td>2012-08-13</td>
      <td>[30, 60)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>472</th>
      <td>2012-11-01</td>
      <td>1</td>
      <td>0</td>
      <td>93541.75</td>
      <td>100000</td>
      <td>645.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - Renewal</td>
      <td>ACH Pull</td>
      <td>9</td>
      <td>0.00</td>
      <td>13328.08</td>
      <td>2012-11-01</td>
      <td>[1, 5)</td>
      <td>[0, 1)</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>473</th>
      <td>2012-11-01</td>
      <td>62</td>
      <td>81</td>
      <td>14697.31</td>
      <td>40000</td>
      <td>583.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>38779.38</td>
      <td>2012-07-23</td>
      <td>[60, inf)</td>
      <td>[60, inf)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>474</th>
      <td>2012-11-01</td>
      <td>39</td>
      <td>58</td>
      <td>15000.00</td>
      <td>15000</td>
      <td>518.0</td>
      <td>Direct</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>5</td>
      <td>0.00</td>
      <td>14469.02</td>
      <td>2012-09-12</td>
      <td>[30, 60)</td>
      <td>[30, 60)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>475</th>
      <td>2012-11-01</td>
      <td>19</td>
      <td>38</td>
      <td>9355.63</td>
      <td>10000</td>
      <td>578.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>ACH Pull</td>
      <td>6</td>
      <td>0.00</td>
      <td>2463.21</td>
      <td>2012-10-04</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
      <td>19</td>
    </tr>
    <tr>
      <th>476</th>
      <td>2012-11-01</td>
      <td>22</td>
      <td>41</td>
      <td>75000.00</td>
      <td>75000</td>
      <td>625.0</td>
      <td>FAP: Managed Application Program</td>
      <td>Loan - New Customer</td>
      <td>Split Funding</td>
      <td>12</td>
      <td>0.00</td>
      <td>14640.79</td>
      <td>2012-11-01</td>
      <td>[10, 30)</td>
      <td>[30, 60)</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
<p>477 rows × 16 columns</p>
</div>




```python
model = sklearn.tree.DecisionTreeRegressor(max_depth=5)
training_data = df[['average_bank_balance__c', 'new_outstanding_principal_balance', 'initial_loan_amount', 'fico', 'term', 'gain']].dropna()
model.fit(training_data[['average_bank_balance__c', 'new_outstanding_principal_balance', 'initial_loan_amount', 'fico', 'term']], training_data['gain'])
```




    DecisionTreeRegressor(criterion='mse', max_depth=5, max_features=None,
               max_leaf_nodes=None, min_impurity_split=1e-07,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, presort=False, random_state=None,
               splitter='best')




```python
model.feature_importances_
```




    array([ 0.01417983,  0.70088777,  0.07805171,  0.20688069,  0.        ])




```python
sklearn.tree.export_graphviz(model, "tree.dot", )
```


```python
transition_matrix(df["days_delinquent_old_bin"], df["days_delinquent_new_bin"])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>days_delinquent_new_bin</th>
      <th>[0, 1)</th>
      <th>[1, 5)</th>
      <th>[5, 10)</th>
      <th>[10, 30)</th>
      <th>[30, 60)</th>
      <th>[60, inf)</th>
    </tr>
    <tr>
      <th>days_delinquent_old_bin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>[0, 1)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[1, 5)</th>
      <td>0.212598</td>
      <td>0.417323</td>
      <td>0.118110</td>
      <td>0.251969</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[5, 10)</th>
      <td>0.060606</td>
      <td>0.030303</td>
      <td>0.378788</td>
      <td>0.530303</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[10, 30)</th>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.016000</td>
      <td>0.376000</td>
      <td>0.528000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[30, 60)</th>
      <td>0.057143</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.466667</td>
      <td>0.476190</td>
    </tr>
    <tr>
      <th>[60, inf)</th>
      <td>0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.833333</td>
    </tr>
  </tbody>
</table>
</div>




```python
transition_matrix(df["days_delinquent_old_bin"], df["days_delinquent_new_bin"], df["new_outstanding_principal_balance"])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>days_delinquent_new_bin</th>
      <th>[0, 1)</th>
      <th>[1, 5)</th>
      <th>[5, 10)</th>
      <th>[10, 30)</th>
      <th>[30, 60)</th>
      <th>[60, inf)</th>
    </tr>
    <tr>
      <th>days_delinquent_old_bin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>[0, 1)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[1, 5)</th>
      <td>0.187963</td>
      <td>0.403151</td>
      <td>0.121824</td>
      <td>0.287061</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[5, 10)</th>
      <td>0.010089</td>
      <td>0.009863</td>
      <td>0.425709</td>
      <td>0.554338</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[10, 30)</th>
      <td>0.016309</td>
      <td>0.000000</td>
      <td>0.016963</td>
      <td>0.425131</td>
      <td>0.541597</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[30, 60)</th>
      <td>0.008393</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.492230</td>
      <td>0.499377</td>
    </tr>
    <tr>
      <th>[60, inf)</th>
      <td>0.074384</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.925616</td>
    </tr>
  </tbody>
</table>
</div>




```python
transition_matrix(df["days_delinquent_old_bin"], df["days_delinquent_new_bin"], df["fico"])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>days_delinquent_new_bin</th>
      <th>[0, 1)</th>
      <th>[1, 5)</th>
      <th>[5, 10)</th>
      <th>[10, 30)</th>
      <th>[30, 60)</th>
      <th>[60, inf)</th>
    </tr>
    <tr>
      <th>days_delinquent_old_bin</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>[0, 1)</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[1, 5)</th>
      <td>0.208215</td>
      <td>0.423840</td>
      <td>0.119934</td>
      <td>0.248010</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[5, 10)</th>
      <td>0.059469</td>
      <td>0.032156</td>
      <td>0.380627</td>
      <td>0.527748</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[10, 30)</th>
      <td>0.080158</td>
      <td>0.000000</td>
      <td>0.015796</td>
      <td>0.380492</td>
      <td>0.523554</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>[30, 60)</th>
      <td>0.054707</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.466289</td>
      <td>0.479004</td>
    </tr>
    <tr>
      <th>[60, inf)</th>
      <td>0.174686</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.825314</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
