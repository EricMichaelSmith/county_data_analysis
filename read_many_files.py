# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 20:10:53 2014

@author: Eric

Reads in many csv and fixed-width files at once.

Sources:
- 2008_to_2012_age_and_sex
    - Searched the US Census' FactFinder with Geographies: {'County': ['All Counties within United States', 'All Counties within United States and Puerto Rico']}
    - Data from S0101: AGE AND SEX
    - 2012 ACS 5-year estimates
    - Downloaded 2014-05-04
- 2008_to_2012_race_and_ethnicity
    - Searched the US Census' FactFinder with Geographies: {'County': ['All Counties within United States', 'All Counties within United States and Puerto Rico']}
    - Data from B03002: HISPANIC OR LATINO ORIGIN BY RACE
    - 2012 ACS 5-year estimates
    - Downloaded 2014-05-04
- 2008_to_2012_social_characteristics
    - Searched the US Census' FactFinder with Geographies: {'County': ['All Counties within United States', 'All Counties within United States and Puerto Rico']}
    - Data from DP02: SELECTED SOCIAL CHARACTERISTICS IN THE UNITED STATES
    - 2012 ACS 5-year estimates
    - Downloaded 2014-05-04 
- 2010_to_2013_population
    - County-level population estimates
    - Data downloaded from http://www.census.gov/popest/data/counties/totals/2013/CO-EST2013-01.html, 2014-05-04
- 2012_income_and_poverty
    - County-level estimates of US poverty and income for 2012
    - Direct link: http://www.census.gov/did/www/saipe/downloads/estmod12/est12ALL.txt
    - Guide to fields can be found http://www.census.gov/did/www/saipe/data/statecounty/data/2012.html
- 2013_area
    - From http://www.census.gov/geo/maps-data/data/gazetteer2013.html; see link for column info
    - Downloaded 2014-05-04
- 2014_health_indicators
    - Available from http://www.countyhealthrankings.org/rankings/data, a program from Robert Wood Johnson
    - The data in the last 10 counties in Wyoming were corrupted, so I had to delete them manually.
- unemployment_statistics
    - From http://www.bls.gov/lau/tables.htm
    - laucntycur14.txt downloaded 2014-09-13, laucntycur13.txt downloaded 2014-09-22, others downloaded 2014-02

Suffixes at the end of variable names:
a: numpy array
b: boolean
d: dictionary
df: pandas DataFrame
l: list
s: string
t: tuple
Underscores indicate chaining: for instance, "foo_t_t" is a tuple of tuples
"""

import os

import config
reload(config)
import utilities
reload(utilities)


def main(con, cur):
    """ Read and transform many data sources needed for the county_data_analysis database. """

    # Fields to extract: first level indicates folder, second level indicates
    # file name, third level indicates fields. In length-2 tuples, index 0
    # represents the field start column for fixed-width files and index 1
    # represents the length of the field.
    field_d = {'2008_to_2012_age_and_sex': 
               {'ACS_12_5YR_S0101_with_ann.csv':
                {'delimiter': ',',
                 'lines_to_ignore': 2,
                 'new_line_string': '\r\n',
                 'null_string': '*****',
                 'total_num_fields': 219,
                 'fields': {2: 'fips_column',
                            179: 'median_age',
                            185: 'sex_ratio'}}},
               '2008_to_2012_race_and_ethnicity':
               {'ACS_12_5YR_B03002_with_ann.csv':
                {'delimiter': ',',
                 'lines_to_ignore': 2,
                 'new_line_string': '\r\n',
                 'null_string': '*****',
                 'total_num_fields': 46,
                 'fields': {2: 'fips_column',
                            5: '2008_to_2012_race_and_ethnicity_total',
                            9: 'white_not_hispanic_number',
                            11: 'black_not_hispanic_number',
                            15: 'asian_not_hispanic_number',
                            27: 'hispanic_number'}}},
               '2008_to_2012_social_characteristics':
               {'ACS_12_5YR_DP02_with_ann.csv':
                {'delimiter': ',',
                 'lines_to_ignore': 2,
                 'new_line_string': '\r\n',
                 'null_string': '(X)',
                 'total_num_fields': 600,
                 'fields': {2: 'fips_column',
                            55: 'households_with_children',
                            59: 'households_with_senior_citizens',
                            61: 'average_household_size',
                            103: 'never_married',
                            119: 'divorced',
                            157: 'fertility',
                            267: 'high_school_graduate',
                            279: 'veterans',
                            319: 'same_house_1_yr_ago',
                            371: 'foreign_born',
                            451: 'language_other_than_english_spoken_at_home'}}},
               '2010_to_2013_population':
               {'PEP_2013_PEPANNRES_with_ann.csv':
                {'delimiter': ',',
                 'lines_to_ignore': 2,
                 'new_line_string': '\r\n',
                 'null_string': None,
                 'total_num_fields': 10,
                 'fields': {2: 'fips_column',
                            5: 'population_2010_census',
                            10: 'population_2013_estimate'}}},
               '2012_income_and_poverty':
               {'est12ALL.txt':
                {'delimiter': None,
                 'lines_to_ignore': 0,
                 'new_line_string': '\r\n',
                 'null_string': None,
                 'fields': {(1, 2): 'fips_state_column',
                            (4, 3): 'fips_county_column',
                            (35, 4): 'in_poverty',
                            (134, 6): 'median_household_income'}}},
               '2013_area':
               {'2013_Gaz_counties_national.txt':
                {'delimiter': r'\t',
                 'lines_to_ignore': 1,
                 'new_line_string': '\r\n',
                 'null_string': None,
                 'total_num_fields': 10,
                 'fields': {2: 'fips_column',
                            7: 'land_area'}}},
               '2014_health_indicators':
               {'2014_CHR_analytic_data.csv':
                {'delimiter': ',',
                 'lines_to_ignore': 2,
                 'new_line_string': '\r\n',
                 'null_string': '',
                 'total_num_fields': 324,
                 'fields': {1: 'fips_state_column',
                            2: 'fips_county_column',
                            6: 'premature_death_rate',
                            31: 'smoking',
                            36: 'obese',
                            71: 'teen_birth_rate',
                            76: 'percent_non_senior_citizens_without_insurance',
                            144: 'violent_crime_rate'}}},
               'unemployment_statistics':
               {'laucnty08.txt':
                {'delimiter': None,
                 'lines_to_ignore': 6,
                 'new_line_string': '\r\n',
                 'null_string': None,
                 'fields': {(19, 2): 'fips_state_column',
                            (26, 3): 'fips_county_column',
                            (129, 4): 'unemployment_rate_2008'}},
                'laucnty12.txt':
                {'delimiter': None,
                 'lines_to_ignore': 6,
                 'new_line_string': '\r\n',
                 'null_string': None,
                 'fields': {(19, 2): 'fips_state_column',
                            (26, 3): 'fips_county_column',
                            (129, 4): 'unemployment_rate_2012'}},
                'laucnty13.txt':
                {'delimiter': None,
                 'lines_to_ignore': 6,
                 'new_line_string': '\r\n',
                 'null_string': None,
                 'fields': {(19, 2): 'fips_state_column',
                            (26, 3): 'fips_county_column',
                            (129, 4): 'unemployment_rate_2013'}}}}
                 
                 
    ## Read in files
    for folder_name in field_d:
        for file_name in field_d[folder_name]:

            # Prepare for creating table
            table_name = file_name.replace('.', '')
            print('Loading {table_name}.'.format(table_name=table_name))
            file_path = os.path.join(config.raw_data_path_s, folder_name,
                                     file_name)
            cur.execute('DROP TABLE IF EXISTS {table_name};'.format(table_name=
                                                                      table_name))
            
            # Create table
            command_s = 'CREATE TABLE {table_name}('
            command_s = command_s.format(table_name=table_name)
            this_table_field_d = field_d[folder_name][file_name]['fields']
            for field in this_table_field_d:
                field_s = '{field_name} FLOAT(16, 5), '
                field_s = field_s.format(field_name=this_table_field_d[field])
                command_s += field_s
            command_s = command_s[:-2] + ');'
            cur.execute(command_s)
            
            
            ## Load all columns
            if field_d[folder_name][file_name]['delimiter']:
                # CSV or tab-delimited tables
            
                # Start command
                command_s = """LOAD DATA LOCAL INFILE '{file_path}'
INTO TABLE {table_name}"""
                command_s = command_s.format(file_path=file_path, table_name=table_name)
                command_s = command_s.replace('\\', r'\\')
                command_s += r"""
FIELDS TERMINATED BY '{delimiter}'
LINES TERMINATED BY '{new_line_string}'
IGNORE {lines_to_ignore} LINES"""
                command_s = command_s.format( \
                    delimiter=field_d[folder_name][file_name]['delimiter'],
                    lines_to_ignore=field_d[folder_name][file_name]['lines_to_ignore'],
                    new_line_string=field_d[folder_name][file_name]['new_line_string'])

                # Add list of fields
                total_num_fields = field_d[folder_name][file_name]['total_num_fields']
                command_s += utilities.construct_field_string(total_num_fields)
                
                # Add a list of field name correspondences: set as NULL if necessary
                command_s += """
SET """
                if field_d[folder_name][file_name]['null_string'] != None:
                    null_string = field_d[folder_name][file_name]['null_string']
                    for field_num, field_name in \
                        field_d[folder_name][file_name]['fields'].iteritems():
                            command_s += """%s = NULLIF(@col%03d, '%s'), """ % \
                                (field_name, field_num, null_string)
                else:
                    for field_num, field_name in \
                        field_d[folder_name][file_name]['fields'].iteritems():
                            command_s += '%s = @col%03d, ' % (field_name, field_num)
                command_s = command_s[:-2] + ';'
                            
            else:
                # Fixed-width tables
            
                # Start command
                command_s = """LOAD DATA LOCAL INFILE '{file_path}'
INTO TABLE {table_name}"""
                command_s = command_s.format(file_path=file_path, table_name=table_name)
                command_s = command_s.replace('\\', r'\\')
                command_s += r"""
LINES TERMINATED BY '\r\n'
IGNORE {lines_to_ignore} LINES
(@whole_row)"""
                command_s = command_s.format( \
                    lines_to_ignore=field_d[folder_name][file_name]['lines_to_ignore'])
                
                # Tell MySQL to add column ranges adding to the fields we want
                # (inspiration: http://stackoverflow.com/questions/11461790/loa
                # ding-fixed-width-space-delimited-txt-file-into-mysql)
                command_s += """
SET """
                for field_num, field_name in \
                    field_d[folder_name][file_name]['fields'].iteritems():
                        command_s += '%s = TRIM(SUBSTR(@whole_row, %d, %d)), ' \
                            % (field_name, field_num[0], field_num[1])
                command_s = command_s[:-2] + ';'
                
#            print(command_s)
            cur.execute(command_s)
                
            
            ## Create proper FIPS column
            if 'fips_column' in field_d[folder_name][file_name]['fields'].values():
                command_s = """ALTER TABLE {table_name}
CHANGE fips_column fips_column INT;""".format(table_name=table_name)
                cur.execute(command_s)
                command_s = """ALTER TABLE {table_name}
CHANGE fips_column {table_name}_fips CHAR(5);""".format(table_name=table_name)
                cur.execute(command_s)
                
            else:
                
                # Cast FIPS columns as CHAR
                command_s = """ALTER TABLE {table_name}
CHANGE fips_state_column fips_state_column INT,
CHANGE fips_county_column fips_county_column INT;""".format(table_name=table_name)
                cur.execute(command_s)
                command_s = """ALTER TABLE {table_name}
CHANGE fips_state_column fips_state_column CHAR(2),
CHANGE fips_county_column fips_county_column CHAR(3);"""
                command_s = command_s.format(table_name=table_name)
                cur.execute(command_s)
                
                # Print columns
#                print('First rows of {table_name}, before padding:'.format(table_name=table_name))
#                cur.execute('SELECT * FROM {table_name};'.format(table_name=table_name))
#                for l_row in range(10):
#                    row = cur.fetchone()
#                    print(row)
                
                # Pad out fips_county_column
                command_s = """UPDATE {table_name}
SET fips_county_column = LPAD(fips_county_column, 3, '0');""".format(table_name=table_name)
                cur.execute(command_s)
                
                # Print columns
#                print('First rows of {table_name}, after padding:'.format(table_name=table_name))
#                cur.execute('SELECT * FROM {table_name};'.format(table_name=table_name))
#                for l_row in range(10):
#                    row = cur.fetchone()
#                    print(row)

                # Concatenate the two FIPS fields
                command_s = 'ALTER TABLE {table_name} ADD {table_name}_fips VARCHAR(5);'
                command_s = command_s.format(table_name=table_name)
                cur.execute(command_s)
                command_s = """UPDATE {table_name}
SET {table_name}_fips = CONCAT(fips_state_column, fips_county_column);"""
                command_s = command_s.format(table_name=table_name)
                cur.execute(command_s)
                
                # Delete unused columns
                command_s = """ALTER TABLE {table_name}
DROP fips_state_column,
DROP fips_county_column;""".format(table_name=table_name)
                cur.execute(command_s)

            
            # Print columns
#            print('First rows of {table_name}:'.format(table_name=table_name))
#            cur.execute('SELECT * FROM {table_name};'.format(table_name=table_name))
#            for l_row in range(10):
#                row = cur.fetchone()
#                print(row)
                
    return field_d