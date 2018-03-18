# competition metric

Explanation of how to calculate the DS Bowl 2018 Competition Metric.

Input: A set of proposed object candidates $A$, a set of true objects $B$, and
a set of thresholds $T$  
Output: DS Bowl 2018 Competition Metric

$prec =$ empty list

for $t \in T$ do:
* $tp = 0$
* $fp = 0$
* $fn = 0$
* $hashit =$ empty mapping from objects to booleans
* for $b \in B$ do:
    * for $a \in A$ do:
        * if $IoU(a, b) > t$
            * $tp++$
            * $hashit(b) = true$
            * $hashit(a) = true$
* for $b \in B$ do:
    * if not $hashit(b)$ then $fn++$
* for $a \in A$ do:
    * if not $hashit(b)$ then $fp++$
* add $tp / (tp + fp + fn)$ to $prec$

return $mean(prec)$
