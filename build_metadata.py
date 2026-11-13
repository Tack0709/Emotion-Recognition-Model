import argparse
import os
import pickle
import re
from collections import defaultdict


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
                    dialogue_id = utterance_id.rsplit('_', 1)[0]
                    speaker_id = utterance_id.rsplit('_', 1)[-1][:1]

                    order.setdefault(dialogue_id, []).append(utterance_id)
                    text_dict[utterance_id] = text.strip()
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

    print(f'Dialogues: {len(order)} | Utterances: {transcript_count} | Speakers: {speaker_ids}')
    if skipped_lines:
        print(f'Ignored non-transcript lines: {skipped_lines}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build IEMOCAP text and dialogue metadata.')
    parser.add_argument('--iemo_root', required=True, help='IEMOCAP_full_release directory')
    parser.add_argument('--output_dir', default='output_data')
    args = parser.parse_args()
    build_metadata(args.iemo_root, args.output_dir)
