# Label order

Every FASTA header stores a 21-bit binary vector in the following fixed order.

| Index | Label |
|---:|---|
| 0 | AAP |
| 1 | ABP |
| 2 | ACP |
| 3 | ACVP |
| 4 | ADP |
| 5 | AEP |
| 6 | AFP |
| 7 | AHIVP |
| 8 | AHP |
| 9 | AIP |
| 10 | AMRSAP |
| 11 | APP |
| 12 | ATP |
| 13 | AVP |
| 14 | BBP |
| 15 | BIP |
| 16 | CPP |
| 17 | DPPIP |
| 18 | QSP |
| 19 | SBP |
| 20 | THP |

Example:

```text
>000000000000000010000
ARRRRCSDRFRNCPADEALCGRRRR
```

The header vector must contain exactly 21 characters and each character must be `0` or `1`.
