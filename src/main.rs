mod huffman;

use std::time::Instant;

struct BenchResult {
    decode_time: f64,
    encode_time: f64,
    file_size: usize,
}

fn bench_huffman(pixels: &[u8], times: usize) -> std::io::Result<BenchResult> {
    let now = Instant::now();
    let mut encoded = Vec::new();
    std::hint::black_box(huffman::encode(&pixels, &mut encoded)?);

    for _ in 1..times {
        let mut encoded = Vec::new();
        std::hint::black_box(huffman::encode(&pixels, &mut encoded)?);
    }

    let encode_time = now.elapsed().as_nanos() as f64 / 1_000_000_f64 / times as f64;

    let now = Instant::now();
    let decoded = std::hint::black_box(huffman::decode(&encoded));
    for _ in 1..times {
        std::hint::black_box(huffman::decode(&encoded));
    }
    let decode_time = now.elapsed().as_nanos() as f64 / 1_000_000_f64 / times as f64;
    assert_eq!(decoded.len(), pixels.len());
    assert!(&decoded == pixels);

    Ok(BenchResult {
        decode_time,
        encode_time,
        file_size: encoded.len(),
    })
}

fn main() -> std::io::Result<()> {
    let mut args = std::env::args();
    let path = args.nth(1).expect("a folder must be provided");
    let times = args
        .next()
        .expect("number of times to run test needs to be provided")
        .parse()
        .expect("valid number of times to be provided");

    for file in std::fs::read_dir(&path)? {
        let file = file?;
        if file.file_name().to_str().unwrap().ends_with(".png") {
            let mut reader = png::Decoder::new(std::fs::File::open(file.path())?).read_info()?;

            let info = reader.info();
            let width = info.width;
            let height = info.height;

            let mut buf = vec![0; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf).unwrap();
            let pixels = &buf[..info.buffer_size()];

            let BenchResult {
                decode_time,
                encode_time,
                file_size,
            } = bench_huffman(pixels, times)?;

            let size = pixels.len();
            let compression_rate = ((size - file_size) as f64 / size as f64) * 100f64;
            let file_size_kb = file_size / 1024;
            let pixel_count = width * height;
            let decode_mpps = pixel_count as f64 / (decode_time as f64 * 1000f64);
            let encode_mpps = pixel_count as f64 / (encode_time as f64 * 1000f64);

            println!(
                "## {path}/{file_name} size: {width}x{height}",
                file_name = file.file_name().to_str().unwrap()
            );
            println!(
                "huffman: {decode_time:.1}\t{encode_time:.1}\t{decode_mpps:.1}\t{encode_mpps:.1}\t{file_size_kb}\t{original}\t{compression_rate:.1}%",
                original = pixels.len() / 1024
            );
            println!("");
        }
    }
    Ok(())
}
