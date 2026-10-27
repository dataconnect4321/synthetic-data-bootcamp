"""The FLIP paper's other datasets (arXiv:2508.21815, Table 2), all with sex
as the protected attribute, for the AIM hybrid and the shared scorer.

  acs_al, acs_ut  ACS-I ("new Adult", Ding et al. 2021): the folktables
                  ACSIncome task (income > $50K) on the 2018 1-year ACS PUMS
                  person files of Alabama and Utah. The filter is folktables'
                  adult_filter: AGEP > 16, PINCP > 100, WKHP > 0, PWGTP >= 1.
                  OCCP (~500 codes) and POBP (~150 codes) are grouped by
                  their published code ranges (occupation major groups,
                  birthplace regions), fixed in advance, so no data-dependent
                  choice enters the release.
  dutch           Dutch census 2001 (Van der Laan 2000), all categorical;
                  target occupation (1 = high-level). Exact duplicate rows
                  are dropped, which reproduces the paper's 18,440 rows.

Files (not committed): Data/acs/psam_p01.csv (AL), Data/acs/psam_p49.csv
(UT), Data/dutch/dutch.csv. Every loader returns raw values with the
categorical columns as strings and the target as '0' / '1'.
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')

ACS_CAT = ['SEX', 'income', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP',
           'RAC1P']
CONFIG = {
    'acs_al': dict(numeric=['AGEP', 'WKHP'], categorical=ACS_CAT,
                   sensitive='SEX', target='income'),
    'acs_ut': dict(numeric=['AGEP', 'WKHP'], categorical=ACS_CAT,
                   sensitive='SEX', target='income'),
    'dutch': dict(numeric=[],
                  categorical=['sex', 'occupation', 'age',
                               'household_position', 'household_size',
                               'prev_residence_place', 'citizenship',
                               'country_birth', 'edu_level',
                               'economic_status', 'cur_eco_activity',
                               'marital_status'],
                  sensitive='sex', target='occupation'),
}

# 2018 Census occupation codes: lower bounds of the major groups
OCCP_GROUPS = [(10, 'MGR'), (500, 'BUS'), (800, 'FIN'), (1005, 'CMM'),
               (1300, 'ENG'), (1600, 'SCI'), (2000, 'CMS'), (2100, 'LGL'),
               (2200, 'EDU'), (2600, 'ENT'), (3000, 'MED'), (3600, 'HLS'),
               (3700, 'PRT'), (4000, 'EAT'), (4200, 'CLN'), (4300, 'PRS'),
               (4700, 'SAL'), (5000, 'OFF'), (6000, 'FFF'), (6200, 'CON'),
               (6800, 'EXT'), (7000, 'RPR'), (7700, 'PRD'), (9000, 'TRN'),
               (9800, 'MIL')]
# place-of-birth codes: US states, US territories, then world regions
POBP_GROUPS = [(1, 'US'), (60, 'US-territory'), (100, 'Europe'),
               (200, 'Asia'), (300, 'Americas'), (400, 'Africa'),
               (500, 'Oceania')]


def _group(codes, groups):
    lows = np.array([g[0] for g in groups])
    names = np.array([g[1] for g in groups], dtype=object)
    return names[np.searchsorted(lows, codes.astype(int), side='right') - 1]


def _acs(state_file):
    cols = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP',
            'SEX', 'RAC1P', 'PINCP', 'PWGTP']
    a = pd.read_csv(os.path.join(DATA_DIR, 'acs', state_file), usecols=cols)
    a = a[(a.AGEP > 16) & (a.PINCP > 100) & (a.WKHP > 0) & (a.PWGTP >= 1)]
    a = a.dropna(subset=cols).reset_index(drop=True)
    df = pd.DataFrame({'AGEP': a.AGEP.astype(int), 'WKHP': a.WKHP.astype(int)})
    df['SEX'] = a.SEX.map({1: 'Male', 2: 'Female'})
    df['income'] = (a.PINCP > 50000).astype(int).astype(str)
    for c in ('COW', 'SCHL', 'MAR', 'RELP', 'RAC1P'):
        df[c] = a[c].astype(int).astype(str)
    df['OCCP'] = _group(a.OCCP, OCCP_GROUPS)
    df['POBP'] = _group(a.POBP, POBP_GROUPS)
    return df


def _dutch():
    d = pd.read_csv(os.path.join(DATA_DIR, 'dutch', 'dutch.csv'))
    d = d.drop_duplicates().reset_index(drop=True)
    return d.astype(str)


LOADERS = {'acs_al': lambda: _acs('psam_p01.csv'),
           'acs_ut': lambda: _acs('psam_p49.csv'),
           'dutch': _dutch}


def load(name):
    cfg = CONFIG[name]
    df = LOADERS[name]()
    return df[cfg['numeric'] + cfg['categorical']]
