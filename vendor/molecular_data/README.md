# Molecular and atomic transition data

`LAMDA/` contains five transition-data files:
`c+.dat`, `catom.dat`, `co.dat`, `hco+.dat`, and `oatom.dat`.

Each file lists energy levels, radiative transitions and collision partners
in the LAMDA format. Transition and collision references are recorded in the
file headers/comments. These data are inputs to DESPOTIC table building;
they are not emission-table outputs or simulation snapshots.

The table builder first honors an explicit `DESPOTIC_HOME` and installed
DESPOTIC molecular data. When neither is available, it uses this directory as
`DESPOTIC_HOME`, whose `LAMDA/` child supplies the files. Existing-table
snapshot processing and plotting do not need these files.
