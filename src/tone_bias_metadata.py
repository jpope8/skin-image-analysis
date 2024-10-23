
import sys
import csv

"""
isic image download --search 'fitzpatrick_skin_type:I OR fitzpatrick_skin_type:II OR fitzpatrick_skin_type:III OR fitzpatrick_skin_type:IV OR fitzpatrick_skin_type:V OR fitzpatrick_skin_type:VI' .
"""
import tone_bias_dataset as dataset

def read_metadata_csv( metadata_filename, record_key=None ):
    attr2index_map = {}
    # self.index2attr_map = {}
    metadata_filename = metadata_filename

    records = {}

    with open(metadata_filename, 'r') as file:
        reader = csv.reader(file)
        line_count = 0
        header_line = None
        for line in reader:
            if header_line is None:
                header_line = line
                # print(header_line)
                index = 0
                for attr in header_line:
                    attr2index_map[attr] = index
                    # self.index2attr_map[index] = attr
                    index += 1
            else:
                record = {}
                for attribute in attr2index_map.keys():
                    index = attr2index_map[attribute]
                    value = line[index]
                    record[attribute] = value.strip()
                # print(f"LINE {line}")  # Output: list of columns in each row
                key = line_count
                if record_key is not None:
                    key = record[record_key]

                # Check to make sure the key is unique
                if key in records:
                    existing_record = records[key]
                    raise ValueError(f"Key {key} already exists in records.")
                    # print(f"DUPLICATE: {key}")
                    # print(f"    existing: {existing_record}")
                    # print(f"      record: {record}")
                records[key] = record
                line_count += 1
    return records

class Dataframe:
    def __init__(self, records, metadata_filename=None):
        self.metadata_filename = metadata_filename
        self.records = records

    def keys(self):
        return self.records.keys()

    def record(self, record_key):
        return self.records[record_key]

    def partition(self, categorical_attribute):
        partitions = dict()
        for record_key in self.keys():
            record = self.record(record_key)
            value = record[categorical_attribute]
            if value not in partitions:
                partitions[value] = dict()
            attribute_records = partitions[value]
            attribute_records[record_key] = record
        return partitions

    def partition_type_diagnosis(self, skin_type, diagnosis):
        records = dict()
        for record_key in self.keys():
            record = self.record(record_key)
            a_skin_type = record["fitzpatrick_skin_type"]
            a_diagnosis = record["benign_malignant"]
            if a_skin_type == skin_type and a_diagnosis == diagnosis:
                records[record_key] = record
        return records

    def __len__(self):
        return len(self.records)


def skin_type(dataframe, skin_type):
    #has_skin_type = 0
    records = {}
    for record_key in dataframe.keys():
        record = dataframe.record(record_key)
        fitzpatrick_skin_type = record['fitzpatrick_skin_type']
        lesion_id = record['lesion_id']
        if fitzpatrick_skin_type == skin_type:
            records[record_key] = record
    return records


def records_subset(records, attribute, value):
    subset = dict()
    for record_key in records.keys():
        record = records[record_key]
        a_value = record[attribute]
        if a_value == value:
            subset[record_key] = record
    return subset

def partition_dataframe(dataframe, categorical_attribute):
    """
    Returns dict of dicts.  For example, categorical_attribute="diagnosis"
    ["malignant"] -> dict of records
    ["benign"] -> dict of records
    :param dataframe:
    :param categorical_attribute:
    :return:
    """
    partitions = {}
    for record_key in dataframe.keys():
        record = dataframe.record(record_key)
        value = record[categorical_attribute]
        if value not in partitions:
            partitions[value] = dict()
        records = partitions[value]
        records[record_key] = record
    return partitions

def print_partition(dataframe, categorical_attribute):
    """
    First partitions dataframe according to categorical_attribute
    and then prints out each type and number of records.
    :param dataframe:
    :param categorical_attribute:
    :return:
    """
    types = partition_dataframe(dataframe, categorical_attribute)
    for atype in types.keys():
        records = types[atype]
        print(f"{categorical_attribute}: {atype} = {len(records)}")

def collection_partition( dataframe ):
    """
    Determine the unique attribution (ie collection).
    Note better to use partition_dataframe(dataframe, "attribution")
    which uses isic_id, whereas here we use lesion_id which
    turns out not to be unique between collections!!!
    :param dataframe:
    :return:
    """
    collections = {}
    for i in dataframe.keys():
        record = dataframe.record(i)
        # print(row)
        attribution = record['attribution']
        if attribution not in collections:
            collections[attribution] = {}
        collection = collections[attribution]
        lesion_id = record['lesion_id']
        collection[lesion_id] = record

    sizeable_collections = 0
    for attribution in collections.keys():
        collection = collections[attribution]
        print(f"{len(collection)}\t{attribution}")
        if len(collection) > 1:
            sizeable_collections += 1
    print(f"Collections {len(collections)}  sizeable {sizeable_collections}")

def partition_skin_type( dataframe ):
    skin_type_1 = skin_type(dataframe, "I")
    print(f"skin_type_1 {len(skin_type_1)}")

    skin_type_2 = skin_type(dataframe, "II")
    print(f"skin_type_2 {len(skin_type_2)}")

    skin_type_3 = skin_type(dataframe, "III")
    print(f"skin_type_3 {len(skin_type_3)}")

    skin_type_4 = skin_type(dataframe, "IV")
    print(f"skin_type_4 {len(skin_type_4)}")

    skin_type_5 = skin_type(dataframe, "V")
    print(f"skin_type_5 {len(skin_type_5)}")

    skin_type_6 = skin_type(dataframe, "VI")
    print(f"skin_type_6 {len(skin_type_6)}")

    skin_type_none = skin_type(dataframe, "")
    print(f"skin_type_none {len(skin_type_none)}")

def main():
    # Issues command ./isic metadata download > metadata.csv
    # Downloading metadata records (482,781)
    attributes = [
        'isic_id',
        'attribution',
        'copyright_license',
        'acquisition_day',
        'age_approx',
        'anatom_site_general',
        'benign_malignant',
        'clin_size_long_diam_mm',
        'concomitant_biopsy',
        'dermoscopic_type',
        'diagnosis',
        'diagnosis_confirm_type',
        'family_hx_mm',
        'fitzpatrick_skin_type',
        'image_type',
        'lesion_id',
        'mel_class',
        'mel_mitotic_index',
        'mel_thick_mm',
        'mel_type',
        'mel_ulcer',
        'melanocytic',
        'nevus_type',
        'patient_id',
        'personal_hx_mm',
        'pixels_x',
        'pixels_y',
        'sex',
        'tbp_tile_type'
        ]



    # Note that lesion_id is not unique and often zero-length string
    # Proper unique key if the isic_id
    metadata_filename = sys.argv[1]
    records = read_metadata_csv(metadata_filename, 'isic_id')
    dataframe = Dataframe(records)
    #dataframe = OldDataframe(metadata_filename, 'isic_id')
    collection_partition(dataframe)
    partition_skin_type(dataframe)
    print(f"Dataframe {len(dataframe)}")

    print_partition(dataframe, "dermoscopic_type")
    print_partition(dataframe, "attribution")
    print_partition(dataframe, "fitzpatrick_skin_type")

    subset = dataframe.partition_type_diagnosis("I","benign")
    print(f"I and benign: {len(subset)}")
    subset = dataframe.partition_type_diagnosis("II", "benign")
    print(f"II and benign: {len(subset)}")


    types = ["I","II","III","IV","V","VI"]
    diagnoses = ["benign", "malignant", "indeterminate/malignant","indeterminate/malignant"]
    for type in types:
        type_records = records_subset(records, "fitzpatrick_skin_type", type)
        print(f"Type {type}:  {len(type_records)}")
        for diagnosis in diagnoses:
            type_diagnosis_records = records_subset(type_records, "benign_malignant", diagnosis)
            print(f"    {type} and {diagnosis} :  {len(type_diagnosis_records)}")

    #type_I_records = records_subset(records, "fitzpatrick_skin_type", "I")
    #print(f"I: {len(type_I_records)}")
    #type_I__benign_records = records_subset(type_I_records, "benign_malignant", "benign")
    #print(f"I and benign: {len(type_I__benign_records)}")
    types = ["light", "dark"]
    diagnoses = ["benign", "malignant"]
    for type in types:
        type_records = records_subset(records, "skin_tone", type)
        print(f"Type {type}:  {len(type_records)}")
        for diagnosis in diagnoses:
            type_diagnosis_records = records_subset(type_records, "benign_malignant", diagnosis)
            print(f"    {type} and {diagnosis} :  {len(type_diagnosis_records)}")


if __name__ == "__main__":
    main()
