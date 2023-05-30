mod huffman;

use std::time::Instant;

struct BenchResult {
    decode_time: f64,
    encode_time: f64,
    file_size: usize,
}

fn bench_huffman<A: huffman::Alphabet + Clone>(
    pixels: &[A],
    times: usize,
) -> std::io::Result<BenchResult> {
    let now = Instant::now();
    let mut encoded = Vec::new();
    std::hint::black_box(huffman::encode(&pixels, &mut encoded)?);

    for _ in 1..times {
        let mut encoded = Vec::new();
        std::hint::black_box(huffman::encode(&pixels, &mut encoded)?);
    }

    let encode_time = now.elapsed().as_nanos() as f64 / 1_000_000_f64 / times as f64;

    let now = Instant::now();
    let decoded = std::hint::black_box(huffman::decode::<A>(&encoded));
    for _ in 1..times {
        std::hint::black_box(huffman::decode::<A>(&encoded));
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

fn bench_pixel<const N: usize>(pixels: &[u8], times: usize) -> std::io::Result<BenchResult> {
    let buf = &unsafe { std::mem::transmute::<_, &[[u8; 3]]>(pixels) }[..pixels.len() / N];
    bench_huffman(buf, times)
}

fn print_results(size: usize, width: u32, height: u32, algo: &str, results: BenchResult) -> f64 {
    let BenchResult {
        decode_time,
        encode_time,
        file_size,
    } = results;

    let compression_rate = ((size as isize - file_size as isize) as f64 / size as f64) * 100f64;
    let pixel_count = width * height;
    let file_size_kb = file_size / 1024;
    let decode_mpps = pixel_count as f64 / (decode_time as f64 * 1000f64);
    let encode_mpps = pixel_count as f64 / (encode_time as f64 * 1000f64);

    println!("{algo:15} {decode_time:.1}\t{encode_time:.1}\t{decode_mpps:.1}\t{encode_mpps:.1}\t{file_size_kb}\t{compression_rate:.1}%");

    compression_rate
}

fn main() -> std::io::Result<()> {
    let mut args = std::env::args();
    let path = args.nth(1).expect("a folder must be provided");
    let times = args
        .next()
        .expect("number of times to run test needs to be provided")
        .parse()
        .expect("valid number of times to be provided");

    let mut u8_size = 0;
    let mut pixel_size = 0;
    let mut total_size = 0;
    let mut count = 0;

    println!("## Benchmarking Huffman {path} with {times} runs");

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
            total_size += info.buffer_size();

            let size = pixels.len();
            let pixel_count = (width * height) as usize;
            let channels = pixels.len() / pixel_count;

            println!(
                "## {path}/{file_name} size: {width}x{height}",
                file_name = file.file_name().to_str().unwrap()
            );

            let result = bench_huffman(pixels, times)?;
            u8_size += result.file_size;
            print_results(size, width, height, "huffman_u8:", result);

            let result = match channels {
                3 => bench_pixel::<3>(pixels, times),
                _ => bench_pixel::<4>(pixels, times),
            }?;
            pixel_size += result.file_size;
            print_results(size, width, height, "huffman_pixel:", result);

            count += 1;
            println!("");
        }
    }

    println!("# Total for {path} {count} images");
    println!("Algorithm      Size KiB\tCompression rate");
    println!("raw:           {}\t0.0%", total_size / 1024);
    println!(
        "huffman_u8:    {}\t{:.1}%",
        u8_size / 1024,
        ((total_size - u8_size) as f64 / total_size as f64) * 100f64
    );
    println!(
        "huffman_pixel: {}\t{:.1}%",
        pixel_size / 1024,
        ((total_size - pixel_size) as f64 / total_size as f64) * 100f64
    );

    Ok(())
}
