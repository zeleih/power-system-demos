# Port definitions

## CEPRI36

The CEPRI36 paper reproduction treats Bus1 through Bus8 as eight aggregated IBR
connection ports. All other active buses are eliminated by Kron reduction.
The capacity-normalization matrix contains the rated MVA values of these eight
aggregated ports on the 100-MVA system base.

This is the explicit benchmark interpretation used by both the PSASP-record and
ANDES workflows.

## IL200 mixed synchronous-machine/IBR system

The IL200 workflow separates measurement ports from final gSCR evaluation
ports:

1. PMU voltage and current are collected at all 49 online source terminals;
2. the external passive-network source-port admittance is identified;
3. 38 synchronous-machine terminals are terminated with Norton admittances
   based on machine-base \(x_d''\), converted to the 100-MVA system base;
4. the final reduced network retains only the 11 REGCA1 IBR buses;
5. the capacity matrix contains only the rated capacities of those 11 IBRs.

The final IBR buses are:

```text
65, 104, 105, 114, 115, 125, 126, 127, 135, 136, 147
```

Thus, synchronous machines contribute network support but are not inserted into
the IBR capacity-normalization matrix.
