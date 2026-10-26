# Baseload district-heat variant

Run this variant with `config/config.baseload-weather.yaml` followed by
`config/config.baseload-chp.yaml`. On the cluster, set `BASELOAD_VARIANT=chp`
when submitting `hpc/baseload_sweep.sbatch`. Results go to
`results/weather-<year>/baseload-chp-free-opex/`, separate from the
electricity-only experiment.

For each AC node with a matching `urban central heat` bus, the baseload plant
may inject district heat. Let `K` be its electricity-only nameplate capacity,
`a` its nuclear-like availability factor, `P_gross` the electricity it would
generate without heat extraction, `Q_heat` the heat injected into the district
heat bus, and `P_net` its net electricity supply. The model imposes

```
P_gross <= a * K
Q_heat <= 0.25 * P_gross
P_net = P_gross - 0.17 * Q_heat
```

The [JRC district-heating report (2012)](https://publications.jrc.ec.europa.eu/repository/bitstream/JRC68846/JRC68846_01.pdf)
uses 0.15 MWh of lost electricity per MWh of heat as a general CHP example and
calculates 0.169 for a coal-fired extraction plant at maximum heat output. The
0.17 assumption rounds the latter figure and is an optimistic transfer to the
generic baseload candidate, not a nuclear-specific measured parameter. A
[CEA study of PWR district heating](https://www.epj-n.org/articles/epjn/pdf/2016/01/epjn150084.pdf)
instead estimates about 0.20 at 120 °C; this remains an important sensitivity.
The 0.25 heat-output cap is a simplified plant-level limit, close to the
232 MWth extraction capacity listed for a 1,000 MWe VVER unit in an
[IAEA Bulletin turbine table](https://www.iaea.org/sites/default/files/31304794649.pdf).
For comparison, an [IAEA Sizewell C study](https://www-pub.iaea.org/MTCD/publications/PDF/TE-2056web.pdf)
describes 200 MWth per 1,760 MWe (about 0.11) for a minimally modified EPR.
The 0.25 cap is therefore an optimistic design choice, not a generic retrofit
limit. Heat temperature, turbine design and plant siting can change both
parameters.

The heat link, plant modification and heat-transport connection are assigned
zero additional cost. It only connects buses belonging to the same model node.
The district-heating demand and its distribution-loss markup remain as built
by PyPSA-Eur. This variant therefore estimates a favourable upper bound on
the heat-coupling benefit, not a complete project business case. If it saves
system cost, the maximum affordable plant CAPEX/OPEX rises by that saving;
including omitted connection costs would reduce that headroom.

Case metrics record gross electricity, net electricity, district heat and
foregone electricity. Nameplate capacity still refers to electricity-only
output, so the 10–60 GW sweep is comparable with the electricity-only case.
