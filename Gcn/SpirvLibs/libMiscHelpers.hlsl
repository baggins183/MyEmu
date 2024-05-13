struct Element {
    float num;
};

//[[vk::binding(11, 0)]] RWStructuredBuffer<Element> fuck;

//extern float myImport(float x);

export float mySum(float a, float b) {

    return a + b;// + fuck[0].num;
}