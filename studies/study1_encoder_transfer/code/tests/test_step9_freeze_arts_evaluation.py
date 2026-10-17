import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts' / '12_freeze_arts_evaluation.py'
spec = importlib.util.spec_from_file_location('step9_v6', MODULE_PATH)
step9 = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = step9
spec.loader.exec_module(step9)


class Step9V6Tests(unittest.TestCase):
    def test_canonical_project_schema_uses_aspect_offsets_for_duplicate_terms(self):
        duplicate_sentence = 'And the food, well the food will keep you coming back.'
        local_rows = [
            {
                'sentence': duplicate_sentence, 'aspect_term': 'food',
                'aspect_from': 23, 'aspect_to': 27, 'label': 'positive',
                'source_instance_id': 'local-second', 'sentence_id': '11484031#1129679#2',
            },
            {
                'sentence': duplicate_sentence, 'aspect_term': 'food',
                'aspect_from': 8, 'aspect_to': 12, 'label': 'positive',
                'source_instance_id': 'local-first', 'sentence_id': '11484031#1129679#2',
            },
            {
                'sentence': 'The service was poor.', 'aspect_term': 'service',
                'aspect_from': 4, 'aspect_to': 11, 'label': 'negative',
                'source_instance_id': 'local-negative', 'sentence_id': 'neg',
            },
            {
                'sentence': 'The decor was average.', 'aspect_term': 'decor',
                'aspect_from': 4, 'aspect_to': 9, 'label': 'neutral',
                'source_instance_id': 'local-neutral', 'sentence_id': 'neu',
            },
        ]
        schema = step9.detect_local_schema(local_rows)
        self.assertEqual(schema.start, 'aspect_from')
        self.assertEqual(schema.end, 'aspect_to')
        official = [
            {
                'sentence': duplicate_sentence, 'term': 'food', 'from': 23, 'to': 27,
                'polarity': 'positive', 'arts_source_id': '11484031#1129679#2_0',
            },
            {
                'sentence': duplicate_sentence, 'term': 'food', 'from': 8, 'to': 12,
                'polarity': 'positive', 'arts_source_id': '11484031#1129679#2_1',
            },
            {
                'sentence': 'The service was poor.', 'term': 'service', 'from': 4, 'to': 11,
                'polarity': 'negative', 'arts_source_id': 'neg',
            },
            {
                'sentence': 'The decor was average.', 'term': 'decor', 'from': 4, 'to': 9,
                'polarity': 'neutral', 'arts_source_id': 'neu',
            },
        ]
        mapping, label_map, audit = step9.align_official_clean_to_local(official, local_rows, schema)
        self.assertEqual(mapping['11484031#1129679#2_0']['source_instance_id'], 'local-second')
        self.assertEqual(mapping['11484031#1129679#2_1']['source_instance_id'], 'local-first')
        self.assertEqual(label_map, {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'})
        self.assertEqual(audit['aligned_count'], 4)

    def test_arts_source_sentence_id(self):
        self.assertEqual(step9.arts_source_sentence_id('456:1_0'), '456:1')
        self.assertEqual(step9.arts_source_sentence_id('11302355#533813#0_3'), '11302355#533813#0')
        self.assertIsNone(step9.arts_source_sentence_id('synthetic'))

    def test_laptop_456_clean_edit_aligns_by_sentence_id_and_target_identity(self):
        # Official SemEval text has "manyLegacy". Released ARTS clean text inserts a
        # space ("many Legacy"), shifting all following offsets by +1.
        local_rows = [
            {
                'sentence_id': '456:1',
                'sentence': 'In fact I still use manyLegacy programs (Appleworks, FileMaker Pro, Quicken, Photoshop etc)!',
                'aspect_term': 'FileMaker Pro', 'aspect_from': 53, 'aspect_to': 66,
                'label': 'neutral', 'source_instance_id': 'local-filemaker',
            },
            {
                'sentence_id': 'n:1', 'sentence': 'bad battery', 'aspect_term': 'battery',
                'aspect_from': 4, 'aspect_to': 11, 'label': 'negative', 'source_instance_id': 'n',
            },
            {
                'sentence_id': 'p:1', 'sentence': 'good screen', 'aspect_term': 'screen',
                'aspect_from': 5, 'aspect_to': 11, 'label': 'positive', 'source_instance_id': 'p',
            },
        ]
        official = [
            {
                'arts_source_id': '456:1_0',
                'sentence': 'In fact I still use many Legacy programs (Appleworks, FileMaker Pro, Quicken, Photoshop etc)!',
                'term': 'FileMaker Pro', 'from': 54, 'to': 67, 'polarity': 'neutral',
            },
            {
                'arts_source_id': 'n:1_0', 'sentence': 'bad battery', 'term': 'battery',
                'from': 4, 'to': 11, 'polarity': 'negative',
            },
            {
                'arts_source_id': 'p:1_0', 'sentence': 'good screen', 'term': 'screen',
                'from': 5, 'to': 11, 'polarity': 'positive',
            },
        ]
        schema = step9.detect_local_schema(local_rows)
        self.assertEqual(schema.sentence_id, 'sentence_id')
        mapping, label_map, audit = step9.align_official_clean_to_local(official, local_rows, schema)
        self.assertEqual(mapping['456:1_0']['source_instance_id'], 'local-filemaker')
        self.assertEqual(label_map['neutral'], 'neutral')
        self.assertIn('456:1_0', audit['clean_text_difference_source_ids'])
        self.assertIn('456:1_0', audit['clean_offset_difference_source_ids'])
        self.assertEqual(audit['mode_counts']['sentence_id_unique_term_label'], 3)

    def test_sentence_id_alignment_uses_occurrence_ordinal_when_same_term_repeats(self):
        sent = 'And the food, well the food will keep you coming back.'
        local_rows = [
            {'sentence_id':'x','sentence':sent,'aspect_term':'food','aspect_from':8,'aspect_to':12,'label':'positive','source_instance_id':'first'},
            {'sentence_id':'x','sentence':sent,'aspect_term':'food','aspect_from':23,'aspect_to':27,'label':'positive','source_instance_id':'second'},
            {'sentence_id':'n','sentence':'bad service','aspect_term':'service','aspect_from':4,'aspect_to':11,'label':'negative','source_instance_id':'n'},
            {'sentence_id':'u','sentence':'plain decor','aspect_term':'decor','aspect_from':6,'aspect_to':11,'label':'neutral','source_instance_id':'u'},
        ]
        official = [
            {'arts_source_id':'x_0','sentence':sent,'term':'food','from':23,'to':27,'polarity':'positive'},
            {'arts_source_id':'x_1','sentence':sent,'term':'food','from':8,'to':12,'polarity':'positive'},
            {'arts_source_id':'n_0','sentence':'bad service','term':'service','from':4,'to':11,'polarity':'negative'},
            {'arts_source_id':'u_0','sentence':'plain decor','term':'decor','from':6,'to':11,'polarity':'neutral'},
        ]
        schema=step9.detect_local_schema(local_rows)
        mapping, _, audit=step9.align_official_clean_to_local(official, local_rows, schema)
        self.assertEqual(mapping['x_0']['source_instance_id'], 'second')
        self.assertEqual(mapping['x_1']['source_instance_id'], 'first')
        self.assertEqual(audit['mode_counts']['sentence_id_exact_span'], 2)

    def test_alignment_collects_failures_in_one_pass(self):
        local_rows = [
            {'sentence_id':'a','sentence':'good battery','aspect_term':'battery','aspect_from':5,'aspect_to':12,'label':'positive','source_instance_id':'a'},
            {'sentence_id':'n','sentence':'bad service','aspect_term':'service','aspect_from':4,'aspect_to':11,'label':'negative','source_instance_id':'n'},
            {'sentence_id':'u','sentence':'plain decor','aspect_term':'decor','aspect_from':6,'aspect_to':11,'label':'neutral','source_instance_id':'u'},
        ]
        official = [
            {'arts_source_id':'missing1_0','sentence':'x','term':'x','from':0,'to':1,'polarity':'positive'},
            {'arts_source_id':'missing2_0','sentence':'y','term':'y','from':0,'to':1,'polarity':'negative'},
            {'arts_source_id':'u_0','sentence':'plain decor','term':'decor','from':6,'to':11,'polarity':'neutral'},
        ]
        schema=step9.detect_local_schema(local_rows)
        with tempfile.TemporaryDirectory() as td:
            report=Path(td)/'failures.json'
            with self.assertRaisesRegex(ValueError, 'failures=2'):
                step9.align_official_clean_to_local(official, local_rows, schema, report)
            data=step9.load_json_no_duplicates(report)
            self.assertEqual(data['failure_count'], 2)
            self.assertEqual({x['arts_source_id'] for x in data['failures']}, {'missing1_0','missing2_0'})

    def test_classify_key(self):
        self.assertEqual(step9.classify_record_key('abc'), ('abc', 'Clean'))
        self.assertEqual(step9.classify_record_key('abc_adv1'), ('abc', 'RevTgt'))
        self.assertEqual(step9.classify_record_key('abc_adv2'), ('abc', 'RevNon'))
        self.assertEqual(step9.classify_record_key('abc_adv3'), ('abc', 'AddDiff'))
        with self.assertRaises(ValueError):
            step9.classify_record_key('abc_adv4')

    def test_clean_span_remains_strict(self):
        rec = {'sentence': 'good battery', 'term': 'battery', 'polarity': 'positive', 'id': 'x', 'from': 5, 'to': 12}
        step9.validate_official_record('x', rec)
        bad = dict(rec, **{'from': 0, 'to': 4})
        with self.assertRaises(ValueError):
            step9.validate_official_record('x', bad)

    def test_split_and_lineage(self):
        data = {
            's1': {'sentence':'good battery','term':'battery','polarity':'positive','id':'s1','from':5,'to':12},
            's1_adv1': {'sentence':'bad battery','term':'battery','polarity':'negative','id':'s1','from':4,'to':11},
            's1_adv2': {'sentence':'good battery but bad screen','term':'battery','polarity':'positive','id':'s1','from':5,'to':12},
            's1_adv3': {'sentence':'good battery, but bad screen','term':'battery','polarity':'positive','id':'s1','from':5,'to':12},
        }
        families = step9.split_official_records(data)
        self.assertEqual([x['arts_record_id'] for x in families['RevTgt']], ['s1_adv1'])
        self.assertEqual(families['RevNon'][0]['arts_source_id'], 's1')
        self.assertEqual(families['AddDiff'][0]['polarity'], 'positive')

    def test_official_bad_pizza_offset_is_preserved_not_rewritten(self):
        source_sentence = "It's a basic pizza joint, not much to look at, but the pizza is what I go for."
        variant_sentence = "It's a basic pizza joint, not much to look at, but the pizza is what I do not go for."
        data = {
            '11350539#680470#2_0': {
                'sentence': source_sentence, 'term': 'pizza', 'polarity': 'positive',
                'id': '11350539#680470#2_0', 'from': 55, 'to': 60,
            },
            '11350539#680470#2_0_adv1': {
                'sentence': variant_sentence, 'term': 'pizza', 'polarity': 'negative',
                'id': '11350539#680470#2_0', 'from': 12, 'to': 17,
            },
        }
        row = step9.split_official_records(data)['RevTgt'][0]
        self.assertEqual((row['from'], row['to']), (12, 17))
        self.assertEqual(row['term'], 'pizza')
        self.assertEqual(row['sentence'][12:17], ' pizz')
        self.assertFalse(row['arts_official_span_exact'])
        self.assertEqual(row['arts_span_status'], 'released_span_mismatch_term_present')
        self.assertEqual((row['arts_diagnostic_resolved_from'], row['arts_diagnostic_resolved_to']), (55, 60))
        self.assertEqual(row['arts_processed_metadata_policy'], 'preserve_official_derivative_verbatim')

    def test_official_warm_pitas_to_pitas_is_preserved(self):
        data = {
            '11351628#404492#3_3': {
                'sentence': 'Creamy appetizers--taramasalata, eggplant salad, and Greek yogurt (with cuccumber, dill, and garlic) taste excellent when on warm pitas.',
                'term': 'warm pitas', 'polarity': 'neutral',
                'id': '11351628#404492#3_3', 'from': 125, 'to': 135,
            },
            '11351628#404492#3_3_adv1': {
                'sentence': 'Creamy appetizers--taramasalata, eggplant salad, and Greek yogurt (with cuccumber, dill, and garlic) taste excellent when on cold pitas.',
                'term': 'pitas', 'polarity': 'neutral',
                'id': '11351628#404492#3_3', 'from': 130, 'to': 135,
            },
        }
        row = step9.split_official_records(data)['RevTgt'][0]
        self.assertEqual(row['term'], 'pitas')
        self.assertEqual((row['from'], row['to']), (130, 135))
        self.assertTrue(row['arts_official_span_exact'])
        self.assertTrue(row['arts_target_term_changed'])
        self.assertEqual(row['arts_term_relation'], 'variant_substring_of_source')

    def test_official_delivary_delivery_inconsistency_is_preserved_and_audited(self):
        data = {
            '11359613#884374#3_0': {
                'sentence': 'Not to sound too negative but be wary of the delivary.',
                'term': 'delivary', 'polarity': 'negative',
                'id': '11359613#884374#3_0', 'from': 45, 'to': 53,
            },
            '11359613#884374#3_0_adv1': {
                'sentence': 'Not to sound too positive but do not be weary of the delivery.',
                'term': 'delivary', 'polarity': 'positive',
                'id': '11359613#884374#3_0', 'from': 53, 'to': 61,
            },
        }
        row = step9.split_official_records(data)['RevTgt'][0]
        self.assertEqual(row['sentence'], 'Not to sound too positive but do not be weary of the delivery.')
        self.assertEqual(row['term'], 'delivary')
        self.assertEqual((row['from'], row['to']), (53, 61))
        self.assertEqual(row['arts_released_span_surface'], 'delivery')
        self.assertFalse(row['arts_variant_term_present_in_sentence'])
        self.assertFalse(row['arts_official_span_exact'])
        self.assertEqual(row['arts_span_status'], 'released_span_surface_differs_term_absent')
        self.assertEqual(row['arts_diagnostic_resolution_rule'], 'released_span_surface_only')

    def test_out_of_bounds_derivative_is_audited_not_mutated(self):
        data = {
            's1': {'sentence':'good battery','term':'battery','polarity':'positive','id':'s1','from':5,'to':12},
            's1_adv1': {'sentence':'bad battery','term':'battery','polarity':'negative','id':'s1','from':999,'to':1005},
        }
        row = step9.split_official_records(data)['RevTgt'][0]
        self.assertEqual((row['from'], row['to']), (999, 1005))
        self.assertFalse(row['arts_official_span_in_bounds'])
        self.assertEqual(row['arts_span_status'], 'released_span_out_of_bounds_term_present')
        self.assertEqual(row['arts_diagnostic_resolution_rule'], 'unique_released_term_occurrence')

    def test_duplicate_json_keys_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'x.json'
            p.write_text('{"a": 1, "a": 2}', encoding='utf-8')
            with self.assertRaises(ValueError):
                step9.load_json_no_duplicates(p)

    def test_adapt_variant_preserves_official_term_and_offsets(self):
        local = {'sentence':'good battery','aspect_term':'battery','label':'positive','from':5,'to':12,'example_id':'local-1', 'source_instance_id':'s1', 'cluster_id':'s1'}
        schema = step9.LocalSchema('sentence','aspect_term','label','from','to','example_id')
        variant = {
            'sentence':'bad delivery','term':'delivary','polarity':'negative','from':4,'to':12,
            'arts_record_id':'s1_adv1','arts_source_id':'s1','arts_family':'RevTgt',
            'arts_official_from':4,'arts_official_to':12,'arts_official_span_in_bounds':True,
            'arts_official_span_exact':False,'arts_released_span_surface':'delivery',
            'arts_source_target_ordinal':0,'arts_source_term':'delivary','arts_variant_term':'delivary',
            'arts_target_term_changed':False,'arts_term_relation':'same','arts_source_term_occurrences':1,
            'arts_variant_term_occurrences':0,'arts_variant_term_present_in_sentence':False,
            'arts_span_status':'released_span_surface_differs_term_absent',
            'arts_diagnostic_resolved_from':4,'arts_diagnostic_resolved_to':12,
            'arts_diagnostic_resolution_rule':'released_span_surface_only',
            'arts_processed_metadata_policy':'preserve_official_derivative_verbatim'
        }
        out = step9.adapt_variant_to_local(local, variant, schema, {'negative':'negative','neutral':'neutral','positive':'positive'})
        self.assertEqual(out['sentence'], 'bad delivery')
        self.assertEqual(out['aspect_term'], 'delivary')
        self.assertEqual((out['from'], out['to']), (4, 12))
        self.assertEqual(out['example_id'], 's1_adv1')

    def test_summary_counts_anomalies(self):
        rows = [
            {'arts_record_id':'a','arts_span_status':'released_exact','arts_official_span_exact':True,'arts_official_span_in_bounds':True,'arts_target_term_changed':False,'arts_variant_term_present_in_sentence':True,'arts_term_relation':'same'},
            {'arts_record_id':'b','arts_span_status':'released_span_surface_differs_term_absent','arts_official_span_exact':False,'arts_official_span_in_bounds':True,'arts_target_term_changed':False,'arts_variant_term_present_in_sentence':False,'arts_term_relation':'same'},
        ]
        s = step9.summarize_span_audit(rows)
        self.assertEqual(s['n_official_span_inexact'], 1)
        self.assertEqual(s['n_variant_term_absent_from_sentence'], 1)
        self.assertEqual(s['variant_term_absent_record_ids'], ['b'])

    def test_model_input_contract_accepts_sentence_aspect_term_label_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            mod=root/'absa_xfer'
            mod.mkdir()
            (mod/'training.py').write_text('''\nclass PairDataset:\n    def __getitem__(self, i):\n        r=self.rows[i]\n        first, second=(r["sentence"], r["aspect_term"])\n        x=self.tokenizer(first, second)\n        x["labels"]=r["label"]\n        return x\n''', encoding='utf-8')
            old=step9.SRC
            step9.SRC=root
            try:
                result=step9.verify_model_input_contract()
            finally:
                step9.SRC=old
            self.assertFalse(result['character_offsets_consumed'])
            self.assertEqual(result['pair_dataset_row_keys'], ['aspect_term','label','sentence'])

    def test_model_input_contract_fails_if_offsets_become_model_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            mod=root/'absa_xfer'
            mod.mkdir()
            (mod/'training.py').write_text('''\nclass PairDataset:\n    def __getitem__(self, i):\n        r=self.rows[i]\n        return r["sentence"], r["aspect_term"], r["label"], r["from"]\n''', encoding='utf-8')
            old=step9.SRC
            step9.SRC=root
            try:
                with self.assertRaises(RuntimeError):
                    step9.verify_model_input_contract()
            finally:
                step9.SRC=old

    def test_term_relation_classifier(self):
        self.assertEqual(step9.classify_term_relation('warm pitas', 'warm pitas'), 'same')
        self.assertEqual(step9.classify_term_relation('warm pitas', 'pitas'), 'variant_substring_of_source')
        self.assertEqual(step9.classify_term_relation('pitas', 'warm pitas'), 'source_substring_of_variant')
        self.assertEqual(step9.classify_term_relation('pizza', 'pie'), 'different')


if __name__ == '__main__':
    unittest.main()
