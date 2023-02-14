use gpu_test::run;

fn main() {
    pollster::block_on(run());
}
