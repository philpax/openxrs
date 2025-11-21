// Vertex data structure
struct VSInput
{
    float3 position : POSITION;
    float3 color : COLOR;
};

struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR;
    uint view_id : SV_ViewID;
};

// Constant buffer for transformation matrices
cbuffer TransformBuffer : register(b0)
{
    float4x4 viewProj[2]; // View-projection matrices for both eyes
    float4x4 model;       // Model transformation matrix
};

// Vertex Shader
VSOutput VSMain(VSInput input, uint viewId : SV_ViewID)
{
    VSOutput output;

    // Transform position: model -> view -> projection
    float4 worldPos = mul(float4(input.position, 1.0), model);
    output.position = mul(worldPos, viewProj[viewId]);
    output.color = input.color;
    output.view_id = viewId;

    return output;
}

// Pixel Shader
float4 PSMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
