import argparse
import os
import pickle
import re


TRANSCRIPT_LINE = re.compile(r'^\s*(\S+)\s+\[[^\]]+\]\s*:\s*(.*)$')


def build_metadata(iemo_root, output_dir='output_data'):
    """Build ordered dialogue IDs, utterance texts, and speaker vocabulary."""
    if not os.path.isdir(iemo_root):
        raise FileNotFoundError(f'IEMOCAP root was not found: {iemo_root}')
    os.makedirs(output_dir, exist_ok=True)

    order = {}
    text_dict = {}
    speakers = set()
    transcript_count = 0
    skipped_lines = 0
    excluded_xx_records = []

    for session_num in range(1, 6):
        transcript_dir = os.path.join(
            iemo_root, f'Session{session_num}', 'dialog', 'transcriptions'
        )
        if not os.path.isdir(transcript_dir):
            raise FileNotFoundError(f'Transcription directory was not found: {transcript_dir}')

        for filename in sorted(os.listdir(transcript_dir)):
            if not filename.endswith('.txt') or filename.startswith('._'):
                continue
            path = os.path.join(transcript_dir, filename)
            transcript_records = []
            with open(path, 'r', encoding='utf-8', errors='replace') as transcript_file:
                for line in transcript_file:
                    match = TRANSCRIPT_LINE.match(line.rstrip('\r\n'))
                    if not match:
                        if line.strip():
                            skipped_lines += 1
                        continue

                    utterance_id, text = match.groups()
                    if not utterance_id.startswith(f'Ses0{session_num}'):
                        continue
                    transcript_records.append({
                        'utterance_id': utterance_id,
                        'text': text.strip(),
                    })

            for index, record in enumerate(transcript_records):
                utterance_id = record['utterance_id']
                text = record['text']
                # Match the previous preprocessing behavior: exclude IDs
                # for unknown/other speakers (for example, FXX3 or MXX0).
                if 'XX' in utterance_id:
                    excluded_xx_records.append({
                        'file': os.path.relpath(path, iemo_root),
                        'dialogue_id': utterance_id.rsplit('_', 1)[0],
                        'excluded': record,
                        'previous': transcript_records[index - 1] if index > 0 else None,
                        'next': transcript_records[index + 1] if index + 1 < len(transcript_records) else None,
                    })
                    continue

                dialogue_id = utterance_id.rsplit('_', 1)[0]
                speaker_id = utterance_id.rsplit('_', 1)[-1][:1]

                order.setdefault(dialogue_id, []).append(utterance_id)
                text_dict[utterance_id] = text
                speakers.add(speaker_id)
                transcript_count += 1

    if not transcript_count:
        raise ValueError('No IEMOCAP transcript utterances were parsed.')

    speaker_ids = sorted(speakers)
    speaker_vocab = {
        'stoi': {speaker: index for index, speaker in enumerate(speaker_ids)},
        'itos': {index: speaker for index, speaker in enumerate(speaker_ids)},
    }

    outputs = {
        'order.pkl': order,
        'text_dict.pkl': text_dict,
        'speaker_vocab.pkl': speaker_vocab,
    }
    for filename, value in outputs.items():
        out_path = os.path.join(output_dir, filename)
        with open(out_path, 'wb') as output_file:
            pickle.dump(value, output_file)
        print(f'Saved {out_path}')

    report_path = os.path.join(output_dir, 'xx_excluded_utterances.txt')
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('IEMOCAP XX-excluded utterance report\n')
        report_file.write(f'Excluded XX utterances: {len(excluded_xx_records)}\n')
        report_file.write('=' * 80 + '\n')
        for index, item in enumerate(excluded_xx_records, start=1):
            report_file.write(f'[{index}] Dialogue: {item["dialogue_id"]}\n')
            report_file.write(f'Transcript file: {item["file"]}\n')
            for label, key in (('Previous', 'previous'), ('Excluded', 'excluded'), ('Next', 'next')):
                record = item[key]
                if record is None:
                    report_file.write(f'{label}: (none)\n')
                else:
                    report_file.write(f'{label} ID: {record["utterance_id"]}\n')
                    report_file.write(f'{label} text: {record["text"]}\n')
            report_file.write('-' * 80 + '\n')
    print(f'Saved {report_path} ({len(excluded_xx_records)} excluded XX utterances)')

    print(f'Dialogues: {len(order)} | Utterances: {transcript_count} | Speakers: {speaker_ids}')
    if skipped_lines:
        print(f'Ignored non-transcript lines: {skipped_lines}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build IEMOCAP text and dialogue metadata.')
    parser.add_argument('--iemo_root', required=True, help='IEMOCAP_full_release directory')
    parser.add_argument('--output_dir', default='output_data')
    args = parser.parse_args()
    build_metadata(args.iemo_root, args.output_dir)
