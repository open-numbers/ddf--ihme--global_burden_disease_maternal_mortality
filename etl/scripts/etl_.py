# coding: utf8

import os
import os.path as osp
import pandas as pd

from ddf_utils.str import to_concept_id, format_float_digits
from ddf_utils.factory import ihme

from functools import partial
from zipfile import ZipFile


source_dir = '../source/cause/mmr_cause_ageall/'
output_dir = '../../'

formatter = partial(format_float_digits, digits=2)

# below are configs for downloader when downloading the source files.
MEASURES = [25]
METRICS = [3]
# AGES = [22]


def load_all_data():
    all_data = []
    for n in os.listdir(source_dir):
        print(n)
        if not n.endswith('.zip'):
            continue
        f = osp.join(source_dir, n)
        zf = ZipFile(f)
        fname = osp.splitext(n)[0] + '.csv'
        data = pd.read_csv(zf.open(fname))
        data = data.drop(['upper', 'lower'], axis=1)

        # double check things
        assert set(data['metric'].unique().tolist()) == set(METRICS)
        assert set(data['measure'].unique().tolist()) == set(METRICS)
        # assert set(data['age'].unique().tolist()) == set(AGES)

        # when metric / measure have only one value, enable this line to decrease memory usage
        # data = data.drop(['metric', 'measure'], axis=1)
        all_data.append(data)
        zf.close()


def serve_datapoints_return_measures(data_full: pd.DataFrame, measure: dict, metric: dict):
    all_measures = []

    groups = data_full.groupby(by=['measure', 'metric'])

    for g in groups.groups:
        name  = measure[g[0]] + ' ' + metric[g[1]]
        # print(name)
        concept = to_concept_id(name)
        all_measures.append((concept, name))

        df = groups.get_group(g)
        df = df.rename(columns={'val': concept})
        cause_groups = df.groupby(by='cause')  # split by cause
        cols = ['location', 'sex', 'age', 'cause', 'year', concept]
        df[concept] = df[concept].map(formatter)
        # if concept == 'mmr_rate':
        #     print(df.sex.unique())
        for g_ in cause_groups.groups:
            df_ = cause_groups.get_group(g_)
            # print(g_)
            # print(df_.age.unique())
            # print(len(df_.year.unique()))
            # print(len(df_.location.unique()))
            cause = 'cause-{}'.format(g_)
            by = ['location', 'sex', 'age', cause, 'year']
            file_name = 'ddf--datapoints--' + concept + '--by--' + '--'.join(by) + '.csv'
            file_path = osp.join(output_dir, file_name)
            df_[cols].sort_values(by=['location', 'sex', 'age', 'year']).to_csv(file_path, index=False)
    return all_measures


def serve_entities(md):
    sexs = md['sex'].copy()
    sexs.columns = ['sex', 'name', 'short_name']
    sexs[['sex', 'name']].to_csv('../../ddf--entities--sex.csv', index=False)

    causes = md['cause'].copy()
    causes.columns = ['cause', 'label', 'name', 'medium_name', 'short_name']
    causes.to_csv('../../ddf--entities--cause.csv', index=False)

    locations = md['location'].copy()
    locations.to_csv('../../ddf--entities--location.csv', index=False)

    ages = md['age'].copy()
    ages = ages.sort_values(by='sort')[['age_id', 'name', 'short_name', 'type']]
    ages.columns = ['age', 'name', 'short_name', 'type']
    ages.to_csv('../../ddf--entities--age.csv', index=False)


def main():
    md = ihme.load_metadata()
    metric = md['metric'].copy()
    measure = md['measure'].copy()

    # datapoints
    all_data = load_all_data()
    data_full = all_data.pop()

    for _ in range(len(all_data)):
        data_full = data_full.append(all_data.pop(), ignore_index=True)

    metric = metric.set_index('metric_id')['name'].to_dict()
    measure = measure.set_index('measure_id')['short_name'].to_dict()

    all_measures = serve_datapoints_return_measures(data_full, measure, metric)

    # entities
    serve_entities(md)

    # concepts
    cont_cdf = pd.DataFrame(all_measures, columns=['concept', 'name'])
    cont_cdf['concept_type'] = 'measure'
    cont_cdf.to_csv('../../ddf--concepts--continuous.csv', index=False)

    dis_cdf = pd.DataFrame([
        ['name', 'Name', 'string'],
        ['short_name', 'Short Name', 'string'],
        ['medium_name', 'Medium Name', 'string'],
        ['long_name', 'Long Name', 'string'],
        ['location', 'Location', 'entity_domain'],
        ['sex', 'Sex', 'entity_domain'],
        ['age', 'Age', 'entity_domain'],
        ['cause', 'Cause', 'entity_domain'],
        ['rei', 'Risk/Etiology/Impairment', 'entity_domain'],
        ['label', 'Label', 'string'],
        ['year', 'Year', 'time'],
        ['type', 'Type', 'string']
    ], columns=['concept', 'name', 'concept_type'])

    dis_cdf.sort_values(by='concept').to_csv('../../ddf--concepts--discrete.csv', index=False)

    print("Done.")
