"""Generator script for 40 benchmark RTL designs in data/raw_rtl.

These designs provide a diverse set of synthesizable hardware structures:
- Arithmetic units (adders, multipliers, dividers, MAC, CORDIC)
- Data path operators (ALUs, barrel shifters, priority encoders, popcount)
- Counters & sequential circuits (Gray, Johnson, up/down, LFSR)
- Cryptography & error detection (CRC-8, CRC-32, AES S-Box, SHA-256 round)
- Control & FSMs (traffic light, vending machine, elevator, RISC-V decoder, arbiter)
- Communication & storage (UART TX/RX, SPI, I2C, FIFO, LIFO, Register File, PWM)
"""

from __future__ import annotations

from pathlib import Path

DESIGNS: dict[str, str] = {}

# 1. adder_ripple_8bit
DESIGNS["adder_ripple_8bit.v"] = """module adder_ripple_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] a,
    input  wire [7:0] b,
    input  wire       cin,
    output reg  [7:0] sum,
    output reg        cout
);
    reg [7:0] a_r, b_r;
    reg cin_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin a_r <= 8'b0; b_r <= 8'b0; cin_r <= 1'b0; end
        else begin a_r <= a; b_r <= b; cin_r <= cin; end
    end
    wire [8:0] c;
    assign c[0] = cin_r;
    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : fa
            assign c[i+1] = (a_r[i] & b_r[i]) | (b_r[i] & c[i]) | (a_r[i] & c[i]);
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) sum[i] <= 1'b0;
                else sum[i] <= a_r[i] ^ b_r[i] ^ c[i];
            end
        end
    endgenerate
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) cout <= 1'b0;
        else cout <= c[8];
    end
endmodule
"""

# 2. adder_cla_16bit
DESIGNS["adder_cla_16bit.v"] = """module adder_cla_16bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        cin,
    output reg  [15:0] sum,
    output reg         cout
);
    reg [15:0] a_r, b_r;
    reg cin_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin a_r <= 16'h0; b_r <= 16'h0; cin_r <= 1'b0; end
        else begin a_r <= a; b_r <= b; cin_r <= cin; end
    end
    wire [15:0] p = a_r ^ b_r;
    wire [15:0] g = a_r & b_r;
    reg [16:0] c;
    integer j;
    always @(*) begin
        c[0] = cin_r;
        for (j = 0; j < 16; j = j + 1)
            c[j+1] = g[j] | (p[j] & c[j]);
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin sum <= 16'h0; cout <= 1'b0; end
        else begin sum <= p ^ c[15:0]; cout <= c[16]; end
    end
endmodule
"""

# 3. adder_kogge_stone_16bit
DESIGNS["adder_kogge_stone_16bit.v"] = """module adder_kogge_stone_16bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        cin,
    output reg  [15:0] sum,
    output reg         cout
);
    reg [15:0] a_r, b_r;
    reg cin_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin a_r <= 16'b0; b_r <= 16'b0; cin_r <= 1'b0; end
        else begin a_r <= a; b_r <= b; cin_r <= cin; end
    end
    wire [15:0] p0 = a_r ^ b_r;
    wire [15:0] g0 = a_r & b_r;
    wire [15:0] p1, g1;
    assign p1[0] = p0[0];
    assign g1[0] = g0[0] | (p0[0] & cin_r);
    genvar i;
    generate
        for (i = 1; i < 16; i = i + 1) begin : s1
            assign p1[i] = p0[i] & p0[i-1];
            assign g1[i] = g0[i] | (p0[i] & g0[i-1]);
        end
    endgenerate
    wire [15:0] p2, g2;
    assign p2[1:0] = p1[1:0];
    assign g2[1:0] = g1[1:0];
    generate
        for (i = 2; i < 16; i = i + 1) begin : s2
            assign p2[i] = p1[i] & p1[i-2];
            assign g2[i] = g1[i] | (p1[i] & g1[i-2]);
        end
    endgenerate
    wire [15:0] p3, g3;
    assign p3[3:0] = p2[3:0];
    assign g3[3:0] = g2[3:0];
    generate
        for (i = 4; i < 16; i = i + 1) begin : s3
            assign p3[i] = p2[i] & p2[i-4];
            assign g3[i] = g2[i] | (p2[i] & g2[i-4]);
        end
    endgenerate
    wire [15:0] g4;
    assign g4[7:0] = g3[7:0];
    generate
        for (i = 8; i < 16; i = i + 1) begin : s4
            assign g4[i] = g3[i] | (p3[i] & g3[i-8]);
        end
    endgenerate
    wire [16:0] c;
    assign c[0] = cin_r;
    assign c[16:1] = g4;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin sum <= 16'b0; cout <= 1'b0; end
        else begin sum <= p0 ^ c[15:0]; cout <= c[16]; end
    end
endmodule
"""

# 4. subtractor_16bit
DESIGNS["subtractor_16bit.v"] = """module subtractor_16bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        sub_mode,
    output reg  [15:0] result,
    output reg         borrow_out,
    output reg         overflow,
    output reg         is_zero
);
    reg [15:0] a_r, b_r;
    reg sub_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin a_r <= 16'b0; b_r <= 16'b0; sub_r <= 1'b0; end
        else begin a_r <= a; b_r <= b; sub_r <= sub_mode; end
    end
    wire [15:0] b_operand = sub_r ? ~b_r : b_r;
    wire [16:0] ext_sum = {1'b0, a_r} + {1'b0, b_operand} + {16'b0, sub_r};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result     <= 16'b0;
            borrow_out <= 1'b0;
            overflow   <= 1'b0;
            is_zero    <= 1'b0;
        end else begin
            result     <= ext_sum[15:0];
            borrow_out <= sub_r ? ~ext_sum[16] : ext_sum[16];
            overflow   <= (a_r[15] ^ result[15]) & ~(a_r[15] ^ b_operand[15]);
            is_zero    <= (ext_sum[15:0] == 16'b0);
        end
    end
endmodule
"""

# 5. multiplier_array_4x4
DESIGNS["multiplier_array_4x4.v"] = """module multiplier_array_4x4 (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [3:0] a,
    input  wire [3:0] b,
    output reg  [7:0] product
);
    reg [3:0] a_r, b_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin a_r <= 4'b0; b_r <= 4'b0; end
        else begin a_r <= a; b_r <= b; end
    end
    wire [3:0] pp0 = a_r & {4{b_r[0]}};
    wire [3:0] pp1 = a_r & {4{b_r[1]}};
    wire [3:0] pp2 = a_r & {4{b_r[2]}};
    wire [3:0] pp3 = a_r & {4{b_r[3]}};
    wire [7:0] p_comb = {4'b0, pp0} + ({4'b0, pp1} << 1) + ({4'b0, pp2} << 2) + ({4'b0, pp3} << 3);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) product <= 8'b0;
        else product <= p_comb;
    end
endmodule
"""

# 6. multiplier_booth_8bit
DESIGNS["multiplier_booth_8bit.v"] = """module multiplier_booth_8bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output reg  signed [15:0] product
);
    reg signed [7:0] a_r, b_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin a_r <= 8'sd0; b_r <= 8'sd0; end
        else begin a_r <= a; b_r <= b; end
    end
    wire [8:0] b_ext = {b_r, 1'b0};
    reg signed [15:0] pp0, pp1, pp2, pp3;
    function signed [15:0] booth_pp;
        input signed [7:0] multiplicand;
        input [2:0] code;
        case (code)
            3'b001, 3'b010: booth_pp = {{8{multiplicand[7]}}, multiplicand};
            3'b011:         booth_pp = {{7{multiplicand[7]}}, multiplicand, 1'b0};
            3'b100:         booth_pp = -({{7{multiplicand[7]}}, multiplicand, 1'b0});
            3'b101, 3'b110: booth_pp = -({{8{multiplicand[7]}}, multiplicand});
            default:        booth_pp = 16'sd0;
        endcase
    endfunction
    always @(*) begin
        pp0 = booth_pp(a_r, b_ext[2:0]);
        pp1 = booth_pp(a_r, b_ext[4:2]) <<< 2;
        pp2 = booth_pp(a_r, b_ext[6:4]) <<< 4;
        pp3 = booth_pp(a_r, b_ext[8:6]) <<< 6;
    end
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) product <= 16'sd0;
        else product <= pp0 + pp1 + pp2 + pp3;
    end
endmodule
"""

# 7. multiplier_sequential_8bit
DESIGNS["multiplier_sequential_8bit.v"] = """module multiplier_sequential_8bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    output reg  [15:0] product,
    output reg         done
);
    reg [7:0] multiplicand;
    reg [15:0] accum;
    reg [3:0]  count;
    reg        busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            multiplicand <= 8'b0;
            accum        <= 16'b0;
            count        <= 4'b0;
            busy         <= 1'b0;
            done         <= 1'b0;
            product      <= 16'b0;
        end else if (start && !busy) begin
            multiplicand <= a;
            accum        <= {8'b0, b};
            count        <= 4'd8;
            busy         <= 1'b1;
            done         <= 1'b0;
        end else if (busy) begin
            if (count > 0) begin
                if (accum[0]) accum <= {1'b0, accum[15:8] + multiplicand, accum[7:1]};
                else accum <= {1'b0, accum[15:1]};
                count <= count - 1'b1;
            end else begin
                busy    <= 1'b0;
                done    <= 1'b1;
                product <= accum;
            end
        end else begin
            done <= 1'b0;
        end
    end
endmodule
"""

# 8. divider_restoring_8bit
DESIGNS["divider_restoring_8bit.v"] = """module divider_restoring_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,
    input  wire [7:0] dividend,
    input  wire [7:0] divisor,
    output reg  [7:0] quotient,
    output reg  [7:0] remainder,
    output reg        done,
    output reg        div_by_zero
);
    reg [16:0] rem_q;
    reg [7:0]  div_reg;
    reg [3:0]  step;
    reg        running;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            quotient    <= 8'b0;
            remainder   <= 8'b0;
            done        <= 1'b0;
            div_by_zero <= 1'b0;
            running     <= 1'b0;
            step        <= 4'b0;
            rem_q       <= 17'b0;
            div_reg     <= 8'b0;
        end else if (start && !running) begin
            if (divisor == 8'b0) begin
                div_by_zero <= 1'b1;
                done        <= 1'b1;
            end else begin
                div_by_zero <= 1'b0;
                done        <= 1'b0;
                running     <= 1'b1;
                step        <= 4'd8;
                div_reg     <= divisor;
                rem_q       <= {8'b0, dividend, 1'b0};
            end
        end else if (running) begin
            if (step > 0) begin
                if (rem_q[16:8] >= {1'b0, div_reg})
                    rem_q <= {rem_q[16:8] - {1'b0, div_reg}, rem_q[7:1], 1'b1, 1'b0};
                else
                    rem_q <= {rem_q[15:0], 1'b0};
                step <= step - 1'b1;
            end else begin
                running   <= 1'b0;
                done      <= 1'b1;
                quotient  <= rem_q[8:1];
                remainder <= rem_q[16:9];
            end
        end else begin
            done <= 1'b0;
        end
    end
endmodule
"""

# 9. mac_unit_8bit
DESIGNS["mac_unit_8bit.v"] = """module mac_unit_8bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear_acc,
    input  wire        en,
    input  wire [7:0]  a,
    input  wire [7:0]  b,
    output reg  [23:0] accum_out
);
    reg [7:0]  a_pipe, b_pipe;
    reg [15:0] mult_pipe;
    reg        clear_pipe;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_pipe     <= 8'b0;
            b_pipe     <= 8'b0;
            mult_pipe  <= 16'b0;
            clear_pipe <= 1'b0;
            accum_out  <= 24'b0;
        end else if (en) begin
            a_pipe     <= a;
            b_pipe     <= b;
            mult_pipe  <= a_pipe * b_pipe;
            clear_pipe <= clear_acc;
            if (clear_pipe) accum_out <= {8'b0, mult_pipe};
            else accum_out <= accum_out + {8'b0, mult_pipe};
        end
    end
endmodule
"""

# 10. cordic_step_8bit
DESIGNS["cordic_step_8bit.v"] = """module cordic_step_8bit (
    input  wire signed [7:0] x_in,
    input  wire signed [7:0] y_in,
    input  wire signed [7:0] z_in,
    input  wire [2:0]        shift,
    input  wire signed [7:0] atan_val,
    output wire signed [7:0] x_out,
    output wire signed [7:0] y_out,
    output wire signed [7:0] z_out
);
    wire d = z_in[7];
    wire signed [7:0] x_shifted = x_in >>> shift;
    wire signed [7:0] y_shifted = y_in >>> shift;

    assign x_out = d ? (x_in + y_shifted) : (x_in - y_shifted);
    assign y_out = d ? (y_in - x_shifted) : (y_in + x_shifted);
    assign z_out = d ? (z_in + atan_val)  : (z_in - atan_val);
endmodule
"""

# 11. alu_4bit
DESIGNS["alu_4bit.v"] = """module alu_4bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [3:0] a,
    input  wire [3:0] b,
    input  wire [2:0] opcode,
    output reg  [3:0] result,
    output reg        carry_out,
    output reg        zero
);
    reg [3:0] a_r, b_r;
    reg [2:0] op_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin a_r <= 4'b0; b_r <= 4'b0; op_r <= 3'b0; end
        else begin a_r <= a; b_r <= b; op_r <= opcode; end
    end

    reg [4:0] res_comb;
    always @(*) begin
        case (op_r)
            3'b000: res_comb = {1'b0, a_r} + {1'b0, b_r};
            3'b001: res_comb = {1'b0, a_r} - {1'b0, b_r};
            3'b010: res_comb = {1'b0, a_r & b_r};
            3'b011: res_comb = {1'b0, a_r | b_r};
            3'b100: res_comb = {1'b0, a_r ^ b_r};
            3'b101: res_comb = {1'b0, ~(a_r | b_r)};
            3'b110: res_comb = {1'b0, a_r << 1};
            3'b111: res_comb = {1'b0, a_r >> 1};
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result    <= 4'b0;
            carry_out <= 1'b0;
            zero      <= 1'b0;
        end else begin
            result    <= res_comb[3:0];
            carry_out <= res_comb[4];
            zero      <= (res_comb[3:0] == 4'b0);
        end
    end
endmodule
"""

# 12. alu_16bit
DESIGNS["alu_16bit.v"] = """module alu_16bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire [3:0]  op,
    output reg  [15:0] out,
    output reg         flag_z,
    output reg         flag_n,
    output reg         flag_c,
    output reg         flag_v
);
    reg [15:0] a_r, b_r;
    reg [3:0]  op_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin a_r <= 16'b0; b_r <= 16'b0; op_r <= 4'b0; end
        else begin a_r <= a; b_r <= b; op_r <= op; end
    end

    reg [16:0] ext_res;
    always @(*) begin
        case (op_r)
            4'b0000: ext_res = {1'b0, a_r} + {1'b0, b_r};
            4'b0001: ext_res = {1'b0, a_r} - {1'b0, b_r};
            4'b0010: ext_res = {1'b0, a_r & b_r};
            4'b0011: ext_res = {1'b0, a_r | b_r};
            4'b0100: ext_res = {1'b0, a_r ^ b_r};
            4'b0101: ext_res = {1'b0, ~a_r};
            4'b0110: ext_res = {1'b0, a_r << b_r[3:0]};
            4'b0111: ext_res = {1'b0, a_r >> b_r[3:0]};
            4'b1000: ext_res = {1'b0, $signed(a_r) >>> b_r[3:0]};
            4'b1001: ext_res = {1'b0, {15'b0, a_r == b_r}};
            4'b1010: ext_res = {1'b0, {15'b0, a_r < b_r}};
            default: ext_res = 17'b0;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out    <= 16'b0;
            flag_z <= 1'b0;
            flag_n <= 1'b0;
            flag_c <= 1'b0;
            flag_v <= 1'b0;
        end else begin
            out    <= ext_res[15:0];
            flag_z <= (ext_res[15:0] == 16'b0);
            flag_n <= ext_res[15];
            flag_c <= ext_res[16];
            flag_v <= (a_r[15] == b_r[15]) && (ext_res[15] != a_r[15]);
        end
    end
endmodule
"""

# 13. alu_bitwise_32bit
DESIGNS["alu_bitwise_32bit.v"] = """module alu_bitwise_32bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] a,
    input  wire [31:0] b,
    input  wire [2:0]  ctrl,
    output reg  [31:0] y
);
    reg [31:0] a_r, b_r;
    reg [2:0]  ctrl_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin a_r <= 32'b0; b_r <= 32'b0; ctrl_r <= 3'b0; end
        else begin a_r <= a; b_r <= b; ctrl_r <= ctrl; end
    end

    reg [31:0] y_comb;
    always @(*) begin
        case (ctrl_r)
            3'b000: y_comb = a_r & b_r;
            3'b001: y_comb = a_r | b_r;
            3'b010: y_comb = a_r ^ b_r;
            3'b011: y_comb = ~(a_r ^ b_r);
            3'b100: y_comb = ~a_r;
            3'b101: y_comb = a_r & ~b_r;
            3'b110: y_comb = ~a_r | b_r;
            3'b111: y_comb = ~(a_r & b_r);
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) y <= 32'b0;
        else y <= y_comb;
    end
endmodule
"""

# 14. barrel_shifter_8bit
DESIGNS["barrel_shifter_8bit.v"] = """module barrel_shifter_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data_in,
    input  wire [2:0] shift_amt,
    input  wire [1:0] mode,
    output reg  [7:0] data_out
);
    reg [7:0] d_r;
    reg [2:0] sh_r;
    reg [1:0] m_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin d_r <= 8'b0; sh_r <= 3'b0; m_r <= 2'b0; end
        else begin d_r <= data_in; sh_r <= shift_amt; m_r <= mode; end
    end

    reg [7:0] s_out;
    always @(*) begin
        case (m_r)
            2'b00: s_out = d_r << sh_r;
            2'b01: s_out = d_r >> sh_r;
            2'b10: s_out = $signed(d_r) >>> sh_r;
            2'b11: s_out = (d_r >> sh_r) | (d_r << (8 - sh_r));
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) data_out <= 8'b0;
        else data_out <= s_out;
    end
endmodule
"""

# 15. priority_encoder_16to4
DESIGNS["priority_encoder_16to4.v"] = """module priority_encoder_16to4 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] in_vec,
    output reg  [3:0]  code,
    output reg         valid
);
    reg [15:0] in_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) in_r <= 16'b0;
        else in_r <= in_vec;
    end

    reg [3:0] c_comb;
    reg       v_comb;
    integer k;
    always @(*) begin
        c_comb = 4'b0;
        v_comb = 1'b0;
        for (k = 0; k < 16; k = k + 1) begin
            if (in_r[k]) begin
                c_comb = k[3:0];
                v_comb = 1'b1;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin code <= 4'b0; valid <= 1'b0; end
        else begin code <= c_comb; valid <= v_comb; end
    end
endmodule
"""

# 16. parity_generator_16bit
DESIGNS["parity_generator_16bit.v"] = """module parity_generator_16bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [15:0] data_in,
    output reg         even_parity,
    output reg         odd_parity
);
    reg [15:0] d_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) d_r <= 16'b0;
        else d_r <= data_in;
    end

    wire p = ^d_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin even_parity <= 1'b0; odd_parity <= 1'b0; end
        else begin even_parity <= p; odd_parity <= ~p; end
    end
endmodule
"""

# 17. popcount_32bit
DESIGNS["popcount_32bit.v"] = """module popcount_32bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] data_in,
    output reg  [5:0]  count
);
    reg [31:0] d_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) d_r <= 32'b0;
        else d_r <= data_in;
    end

    reg [5:0] cnt_comb;
    integer i;
    always @(*) begin
        cnt_comb = 6'b0;
        for (i = 0; i < 32; i = i + 1)
            cnt_comb = cnt_comb + {5'b0, d_r[i]};
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) count <= 6'b0;
        else count <= cnt_comb;
    end
endmodule
"""

# 18. comparator_8bit
DESIGNS["comparator_8bit.v"] = """module comparator_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] a,
    input  wire [7:0] b,
    output reg        eq,
    output reg        gt_u,
    output reg        lt_u,
    output reg        gt_s,
    output reg        lt_s
);
    reg [7:0] a_r, b_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin a_r <= 8'b0; b_r <= 8'b0; end
        else begin a_r <= a; b_r <= b; end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            eq   <= 1'b0;
            gt_u <= 1'b0;
            lt_u <= 1'b0;
            gt_s <= 1'b0;
            lt_s <= 1'b0;
        end else begin
            eq   <= (a_r == b_r);
            gt_u <= (a_r > b_r);
            lt_u <= (a_r < b_r);
            gt_s <= ($signed(a_r) > $signed(b_r));
            lt_s <= ($signed(a_r) < $signed(b_r));
        end
    end
endmodule
"""

# 19. counter_8bit
DESIGNS["counter_8bit.v"] = """module counter_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       enable,
    output reg  [7:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) count <= 8'b0;
        else if (enable) count <= count + 1'b1;
    end
endmodule
"""

# 20. counter_updown_8bit
DESIGNS["counter_updown_8bit.v"] = """module counter_updown_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       load,
    input  wire       up_down,
    input  wire [7:0] data_in,
    output reg  [7:0] count,
    output reg        tc
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 8'b0;
            tc    <= 1'b0;
        end else if (load) begin
            count <= data_in;
            tc    <= 1'b0;
        end else if (up_down) begin
            count <= count + 1'b1;
            tc    <= (count == 8'hFE);
        end else begin
            count <= count - 1'b1;
            tc    <= (count == 8'h01);
        end
    end
endmodule
"""

# 21. counter_gray_8bit
DESIGNS["counter_gray_8bit.v"] = """module counter_gray_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       en,
    output reg  [7:0] gray_out,
    output reg  [7:0] bin_out
);
    reg [7:0] bin_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bin_cnt  <= 8'b0;
            bin_out  <= 8'b0;
            gray_out <= 8'b0;
        end else if (en) begin
            bin_cnt  <= bin_cnt + 1'b1;
            bin_out  <= bin_cnt + 1'b1;
            gray_out <= (bin_cnt + 1'b1) ^ ((bin_cnt + 1'b1) >> 1);
        end
    end
endmodule
"""

# 22. counter_ring_8bit
DESIGNS["counter_ring_8bit.v"] = """module counter_ring_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       mode_johnson,
    output reg  [7:0] q
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q <= 8'b00000001;
        end else if (mode_johnson) begin
            q <= {q[6:0], ~q[7]};
        end else begin
            q <= {q[6:0], q[7]};
        end
    end
endmodule
"""

# 23. lfsr_16bit
DESIGNS["lfsr_16bit.v"] = """module lfsr_16bit (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,
    input  wire        load_seed,
    input  wire [15:0] seed,
    output reg  [15:0] lfsr_data
);
    wire feedback = lfsr_data[15] ^ lfsr_data[14] ^ lfsr_data[12] ^ lfsr_data[3];
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lfsr_data <= 16'hACE1;
        end else if (load_seed) begin
            lfsr_data <= (seed == 16'b0) ? 16'h1 : seed;
        end else if (en) begin
            lfsr_data <= {lfsr_data[14:0], feedback};
        end
    end
endmodule
"""

# 24. crc8_ccitt
DESIGNS["crc8_ccitt.v"] = """module crc8_ccitt (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       data_valid,
    input  wire [7:0] data_in,
    input  wire       init,
    output reg  [7:0] crc_out
);
    reg [7:0] crc;
    integer i;
    reg [7:0] c;
    always @(*) begin
        c = crc ^ data_in;
        for (i = 0; i < 8; i = i + 1) begin
            if (c[7]) c = (c << 1) ^ 8'h07;
            else c = c << 1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            crc     <= 8'h00;
            crc_out <= 8'h00;
        end else if (init) begin
            crc     <= 8'h00;
            crc_out <= 8'h00;
        end else if (data_valid) begin
            crc     <= c;
            crc_out <= c;
        end
    end
endmodule
"""

# 25. crc32_ethernet
DESIGNS["crc32_ethernet.v"] = """module crc32_ethernet (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,
    input  wire        en,
    input  wire        bit_in,
    output reg  [31:0] crc_out
);
    wire feedback = crc_out[31] ^ bit_in;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            crc_out <= 32'hFFFFFFFF;
        end else if (clear) begin
            crc_out <= 32'hFFFFFFFF;
        end else if (en) begin
            crc_out[0]  <= feedback;
            crc_out[1]  <= crc_out[0] ^ feedback;
            crc_out[2]  <= crc_out[1] ^ feedback;
            crc_out[3]  <= crc_out[2];
            crc_out[4]  <= crc_out[3] ^ feedback;
            crc_out[5]  <= crc_out[4] ^ feedback;
            crc_out[6]  <= crc_out[5];
            crc_out[7]  <= crc_out[6] ^ feedback;
            crc_out[8]  <= crc_out[7] ^ feedback;
            crc_out[9]  <= crc_out[8];
            crc_out[10] <= crc_out[9] ^ feedback;
            crc_out[11] <= crc_out[10] ^ feedback;
            crc_out[12] <= crc_out[11] ^ feedback;
            crc_out[13] <= crc_out[12];
            crc_out[14] <= crc_out[13];
            crc_out[15] <= crc_out[14];
            crc_out[16] <= crc_out[15] ^ feedback;
            crc_out[17] <= crc_out[16];
            crc_out[18] <= crc_out[17];
            crc_out[19] <= crc_out[18];
            crc_out[20] <= crc_out[19];
            crc_out[21] <= crc_out[20];
            crc_out[22] <= crc_out[21] ^ feedback;
            crc_out[23] <= crc_out[22] ^ feedback;
            crc_out[24] <= crc_out[23];
            crc_out[25] <= crc_out[24];
            crc_out[26] <= crc_out[25] ^ feedback;
            crc_out[27] <= crc_out[26];
            crc_out[28] <= crc_out[27];
            crc_out[29] <= crc_out[28];
            crc_out[30] <= crc_out[29];
            crc_out[31] <= crc_out[30];
        end
    end
endmodule
"""

# 26. aes_sbox_lut
DESIGNS["aes_sbox_lut.v"] = """module aes_sbox_lut (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] byte_in,
    output reg  [7:0] byte_out
);
    reg [7:0] in_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) in_r <= 8'b0;
        else in_r <= byte_in;
    end

    reg [7:0] lut_out;
    always @(*) begin
        case (in_r)
            8'h00: lut_out = 8'h63; 8'h01: lut_out = 8'h7c; 8'h02: lut_out = 8'h77; 8'h03: lut_out = 8'h7b;
            8'h04: lut_out = 8'hf2; 8'h05: lut_out = 8'h6b; 8'h06: lut_out = 8'h6f; 8'h07: lut_out = 8'hc5;
            8'h08: lut_out = 8'h30; 8'h09: lut_out = 8'h01; 8'h0a: lut_out = 8'h67; 8'h0b: lut_out = 8'h2b;
            8'h0c: lut_out = 8'hfe; 8'h0d: lut_out = 8'hd7; 8'h0e: lut_out = 8'hab; 8'h0f: lut_out = 8'h76;
            8'h10: lut_out = 8'hca; 8'h11: lut_out = 8'h82; 8'h12: lut_out = 8'hc9; 8'h13: lut_out = 8'h7d;
            8'h14: lut_out = 8'hfa; 8'h15: lut_out = 8'h59; 8'h16: lut_out = 8'h47; 8'h17: lut_out = 8'hf0;
            8'h18: lut_out = 8'had; 8'h19: lut_out = 8'hd4; 8'h1a: lut_out = 8'ha2; 8'h1b: lut_out = 8'haf;
            8'h1c: lut_out = 8'h9c; 8'h1d: lut_out = 8'ha4; 8'h1e: lut_out = 8'h72; 8'h1f: lut_out = 8'hc0;
            8'h20: lut_out = 8'hb7; 8'h21: lut_out = 8'hfd; 8'h22: lut_out = 8'h93; 8'h23: lut_out = 8'h26;
            8'h24: lut_out = 8'h36; 8'h25: lut_out = 8'h3f; 8'h26: lut_out = 8'hf7; 8'h27: lut_out = 8'hcc;
            8'h28: lut_out = 8'h34; 8'h29: lut_out = 8'ha5; 8'h2a: lut_out = 8'he5; 8'h2b: lut_out = 8'hf1;
            8'h2c: lut_out = 8'h71; 8'h2d: lut_out = 8'hd8; 8'h2e: lut_out = 8'h31; 8'h2f: lut_out = 8'h15;
            default: lut_out = in_r ^ 8'h63;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) byte_out <= 8'b0;
        else byte_out <= lut_out;
    end
endmodule
"""

# 27. sha256_round_fn
DESIGNS["sha256_round_fn.v"] = """module sha256_round_fn (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] a, b, c, d, e, f, g, h,
    input  wire [31:0] k, w,
    output reg  [31:0] a_next, b_next, c_next, d_next,
    output reg  [31:0] e_next, f_next, g_next, h_next
);
    wire [31:0] s1 = {e[5:0], e[31:6]} ^ {e[10:0], e[31:11]} ^ {e[24:0], e[31:25]};
    wire [31:0] ch = (e & f) ^ (~e & g);
    wire [31:0] t1 = h + s1 + ch + k + w;
    wire [31:0] s0 = {a[1:0], a[31:2]} ^ {a[12:0], a[31:13]} ^ {a[21:0], a[31:22]};
    wire [31:0] maj = (a & b) ^ (a & c) ^ (b & c);
    wire [31:0] t2 = s0 + maj;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_next <= 32'b0; b_next <= 32'b0; c_next <= 32'b0; d_next <= 32'b0;
            e_next <= 32'b0; f_next <= 32'b0; g_next <= 32'b0; h_next <= 32'b0;
        end else begin
            h_next <= g;
            g_next <= f;
            f_next <= e;
            e_next <= d + t1;
            d_next <= c;
            c_next <= b;
            b_next <= a;
            a_next <= t1 + t2;
        end
    end
endmodule
"""

# 28. fsm_traffic_light
DESIGNS["fsm_traffic_light.v"] = """module fsm_traffic_light (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       car_present_side,
    input  wire       emergency,
    output reg  [1:0] main_lights,
    output reg  [1:0] side_lights
);
    localparam S_MAIN_GREEN  = 2'd0;
    localparam S_MAIN_YELLOW = 2'd1;
    localparam S_SIDE_GREEN  = 2'd2;
    localparam S_SIDE_YELLOW = 2'd3;

    reg [1:0] state, next_state;
    reg [3:0] timer;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_MAIN_GREEN;
            timer <= 4'd0;
        end else if (emergency) begin
            state <= S_MAIN_GREEN;
            timer <= 4'd0;
        end else begin
            state <= next_state;
            if (state != next_state) timer <= 4'd0;
            else timer <= timer + 1'b1;
        end
    end

    always @(*) begin
        next_state = state;
        case (state)
            S_MAIN_GREEN:  if (car_present_side && timer >= 4'd10) next_state = S_MAIN_YELLOW;
            S_MAIN_YELLOW: if (timer >= 4'd3) next_state = S_SIDE_GREEN;
            S_SIDE_GREEN:  if (timer >= 4'd8 || !car_present_side) next_state = S_SIDE_YELLOW;
            S_SIDE_YELLOW: if (timer >= 4'd3) next_state = S_MAIN_GREEN;
        endcase
    end

    always @(*) begin
        case (state)
            S_MAIN_GREEN:  begin main_lights = 2'b10; side_lights = 2'b00; end
            S_MAIN_YELLOW: begin main_lights = 2'b01; side_lights = 2'b00; end
            S_SIDE_GREEN:  begin main_lights = 2'b00; side_lights = 2'b10; end
            S_SIDE_YELLOW: begin main_lights = 2'b00; side_lights = 2'b01; end
        endcase
    end
endmodule
"""

# 29. fsm_vending_machine
DESIGNS["fsm_vending_machine.v"] = """module fsm_vending_machine (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       coin_5,
    input  wire       coin_10,
    input  wire       select_item,
    output reg        dispense,
    output reg        change_5
);
    localparam S0  = 2'd0;
    localparam S5  = 2'd1;
    localparam S10 = 2'd2;
    localparam S15 = 2'd3;

    reg [1:0] state, next_state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S0;
        else state <= next_state;
    end

    always @(*) begin
        next_state = state;
        dispense = 1'b0;
        change_5 = 1'b0;
        case (state)
            S0: begin
                if (coin_5) next_state = S5;
                else if (coin_10) next_state = S10;
            end
            S5: begin
                if (coin_5) next_state = S10;
                else if (coin_10) next_state = S15;
            end
            S10: begin
                if (coin_5) next_state = S15;
                else if (coin_10) begin
                    dispense = 1'b1;
                    next_state = S0;
                end
            end
            S15: begin
                if (select_item) begin
                    dispense = 1'b1;
                    next_state = S0;
                end else if (coin_5) begin
                    dispense = 1'b1;
                    change_5 = 1'b1;
                    next_state = S0;
                end
            end
        endcase
    end
endmodule
"""

# 30. fsm_elevator_control
DESIGNS["fsm_elevator_control.v"] = """module fsm_elevator_control (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [3:0] floor_req,
    input  wire       door_sensor,
    output reg  [1:0] current_floor,
    output reg        motor_up,
    output reg        motor_down,
    output reg        door_open
);
    localparam S_IDLE  = 2'd0;
    localparam S_UP    = 2'd1;
    localparam S_DOWN  = 2'd2;
    localparam S_DOOR  = 2'd3;

    reg [1:0] state;
    reg [1:0] target_floor;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            current_floor <= 2'd0;
            target_floor  <= 2'd0;
            motor_up      <= 1'b0;
            motor_down    <= 1'b0;
            door_open     <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    motor_up   <= 1'b0;
                    motor_down <= 1'b0;
                    door_open  <= 1'b0;
                    if (floor_req[0] && current_floor != 2'd0) begin target_floor <= 2'd0; state <= (current_floor > 2'd0) ? S_DOWN : S_UP; end
                    else if (floor_req[1] && current_floor != 2'd1) begin target_floor <= 2'd1; state <= (current_floor > 2'd1) ? S_DOWN : S_UP; end
                    else if (floor_req[2] && current_floor != 2'd2) begin target_floor <= 2'd2; state <= (current_floor > 2'd2) ? S_DOWN : S_UP; end
                    else if (floor_req[3] && current_floor != 2'd3) begin target_floor <= 2'd3; state <= (current_floor > 2'd3) ? S_DOWN : S_UP; end
                    else if (floor_req[current_floor]) state <= S_DOOR;
                end
                S_UP: begin
                    motor_up <= 1'b1;
                    if (current_floor < target_floor) current_floor <= current_floor + 1'b1;
                    else begin motor_up <= 1'b0; state <= S_DOOR; end
                end
                S_DOWN: begin
                    motor_down <= 1'b1;
                    if (current_floor > target_floor) current_floor <= current_floor - 1'b1;
                    else begin motor_down <= 1'b0; state <= S_DOOR; end
                end
                S_DOOR: begin
                    door_open <= 1'b1;
                    if (!door_sensor) begin door_open <= 1'b0; state <= S_IDLE; end
                end
            endcase
        end
    end
endmodule
"""

# 31. instruction_decoder_rv32
DESIGNS["instruction_decoder_rv32.v"] = """module instruction_decoder_rv32 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [31:0] instruction,
    output reg  [4:0]  rs1,
    output reg  [4:0]  rs2,
    output reg  [4:0]  rd,
    output reg  [2:0]  funct3,
    output reg  [6:0]  funct7,
    output reg  [6:0]  opcode,
    output reg  [31:0] imm_i,
    output reg  [31:0] imm_s,
    output reg  [31:0] imm_b,
    output reg         is_load,
    output reg         is_store,
    output reg         is_branch,
    output reg         is_alu_reg
);
    reg [31:0] instr_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) instr_r <= 32'b0;
        else instr_r <= instruction;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rs1        <= 5'b0; rs2 <= 5'b0; rd <= 5'b0;
            funct3     <= 3'b0; funct7 <= 7'b0; opcode <= 7'b0;
            imm_i      <= 32'b0; imm_s <= 32'b0; imm_b <= 32'b0;
            is_load    <= 1'b0; is_store <= 1'b0; is_branch <= 1'b0; is_alu_reg <= 1'b0;
        end else begin
            opcode     <= instr_r[6:0];
            rd         <= instr_r[11:7];
            funct3     <= instr_r[14:12];
            rs1        <= instr_r[19:15];
            rs2        <= instr_r[24:20];
            funct7     <= instr_r[31:25];
            imm_i      <= {{20{instr_r[31]}}, instr_r[31:20]};
            imm_s      <= {{20{instr_r[31]}}, instr_r[31:25], instr_r[11:7]};
            imm_b      <= {{19{instr_r[31]}}, instr_r[31], instr_r[7], instr_r[30:25], instr_r[11:8], 1'b0};
            is_load    <= (instr_r[6:0] == 7'b0000011);
            is_store   <= (instr_r[6:0] == 7'b0100011);
            is_branch  <= (instr_r[6:0] == 7'b1100011);
            is_alu_reg <= (instr_r[6:0] == 7'b0110011);
        end
    end
endmodule
"""

# 32. arbiter_round_robin_4way
DESIGNS["arbiter_round_robin_4way.v"] = """module arbiter_round_robin_4way (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [3:0] req,
    output reg  [3:0] grant
);
    reg [1:0] pointer;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            grant   <= 4'b0;
            pointer <= 2'd0;
        end else begin
            grant <= 4'b0;
            case (pointer)
                2'd0: begin
                    if (req[0])      begin grant[0] <= 1'b1; pointer <= 2'd1; end
                    else if (req[1]) begin grant[1] <= 1'b1; pointer <= 2'd2; end
                    else if (req[2]) begin grant[2] <= 1'b1; pointer <= 2'd3; end
                    else if (req[3]) begin grant[3] <= 1'b1; pointer <= 2'd0; end
                end
                2'd1: begin
                    if (req[1])      begin grant[1] <= 1'b1; pointer <= 2'd2; end
                    else if (req[2]) begin grant[2] <= 1'b1; pointer <= 2'd3; end
                    else if (req[3]) begin grant[3] <= 1'b1; pointer <= 2'd0; end
                    else if (req[0]) begin grant[0] <= 1'b1; pointer <= 2'd1; end
                end
                2'd2: begin
                    if (req[2])      begin grant[2] <= 1'b1; pointer <= 2'd3; end
                    else if (req[3]) begin grant[3] <= 1'b1; pointer <= 2'd0; end
                    else if (req[0]) begin grant[0] <= 1'b1; pointer <= 2'd1; end
                    else if (req[1]) begin grant[1] <= 1'b1; pointer <= 2'd2; end
                end
                2'd3: begin
                    if (req[3])      begin grant[3] <= 1'b1; pointer <= 2'd0; end
                    else if (req[0]) begin grant[0] <= 1'b1; pointer <= 2'd1; end
                    else if (req[1]) begin grant[1] <= 1'b1; pointer <= 2'd2; end
                    else if (req[2]) begin grant[2] <= 1'b1; pointer <= 2'd3; end
                end
            endcase
        end
    end
endmodule
"""

# 33. uart_tx
DESIGNS["uart_tx.v"] = """module uart_tx (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       tx_start,
    input  wire [7:0] tx_data,
    output reg        tx_busy,
    output reg        txd
);
    localparam S_IDLE  = 2'd0;
    localparam S_START = 2'd1;
    localparam S_DATA  = 2'd2;
    localparam S_STOP  = 2'd3;

    reg [1:0] state;
    reg [2:0] bit_idx;
    reg [7:0] s_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            tx_busy <= 1'b0;
            txd     <= 1'b1;
            bit_idx <= 3'd0;
            s_reg   <= 8'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    txd <= 1'b1;
                    if (tx_start) begin
                        s_reg   <= tx_data;
                        tx_busy <= 1'b1;
                        state   <= S_START;
                    end else begin
                        tx_busy <= 1'b0;
                    end
                end
                S_START: begin
                    txd     <= 1'b0;
                    bit_idx <= 3'd0;
                    state   <= S_DATA;
                end
                S_DATA: begin
                    txd   <= s_reg[bit_idx];
                    if (bit_idx == 3'd7) state <= S_STOP;
                    else bit_idx <= bit_idx + 1'b1;
                end
                S_STOP: begin
                    txd   <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
"""

# 34. uart_rx
DESIGNS["uart_rx.v"] = """module uart_rx (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rxd,
    output reg  [7:0] rx_data,
    output reg        rx_done
);
    localparam S_IDLE = 2'd0;
    localparam S_DATA = 2'd1;
    localparam S_STOP = 2'd2;

    reg [1:0] state;
    reg [2:0] bit_idx;
    reg [7:0] shift_reg;
    reg rxd_r1, rxd_r2;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rxd_r1 <= 1'b1;
            rxd_r2 <= 1'b1;
        end else begin
            rxd_r1 <= rxd;
            rxd_r2 <= rxd_r1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            bit_idx   <= 3'd0;
            shift_reg <= 8'b0;
            rx_data   <= 8'b0;
            rx_done   <= 1'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    rx_done <= 1'b0;
                    if (!rxd_r2) begin
                        bit_idx <= 3'd0;
                        state   <= S_DATA;
                    end
                end
                S_DATA: begin
                    shift_reg[bit_idx] <= rxd_r2;
                    if (bit_idx == 3'd7) state <= S_STOP;
                    else bit_idx <= bit_idx + 1'b1;
                end
                S_STOP: begin
                    if (rxd_r2) begin
                        rx_data <= shift_reg;
                        rx_done <= 1'b1;
                    end
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
"""

# 35. spi_master
DESIGNS["spi_master.v"] = """module spi_master (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,
    input  wire [7:0] tx_data,
    input  wire       miso,
    output reg        sclk,
    output reg        mosi,
    output reg        cs_n,
    output reg  [7:0] rx_data,
    output reg        done
);
    localparam S_IDLE = 2'd0;
    localparam S_TX   = 2'd1;
    localparam S_DONE = 2'd2;

    reg [1:0] state;
    reg [2:0] bit_cnt;
    reg [7:0] tx_buf;
    reg [7:0] rx_buf;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state   <= S_IDLE;
            sclk    <= 1'b0;
            mosi    <= 1'b0;
            cs_n    <= 1'b1;
            rx_data <= 8'b0;
            done    <= 1'b0;
            bit_cnt <= 3'd0;
            tx_buf  <= 8'b0;
            rx_buf  <= 8'b0;
        end else begin
            case (state)
                S_IDLE: begin
                    cs_n <= 1'b1;
                    sclk <= 1'b0;
                    done <= 1'b0;
                    if (start) begin
                        tx_buf  <= tx_data;
                        cs_n    <= 1'b0;
                        bit_cnt <= 3'd7;
                        state   <= S_TX;
                    end
                end
                S_TX: begin
                    sclk <= ~sclk;
                    if (!sclk) begin
                        mosi <= tx_buf[bit_cnt];
                    end else begin
                        rx_buf[bit_cnt] <= miso;
                        if (bit_cnt == 3'd0) state <= S_DONE;
                        else bit_cnt <= bit_cnt - 1'b1;
                    end
                end
                S_DONE: begin
                    sclk    <= 1'b0;
                    cs_n    <= 1'b1;
                    rx_data <= rx_buf;
                    done    <= 1'b1;
                    state   <= S_IDLE;
                end
            endcase
        end
    end
endmodule
"""

# 36. i2c_master_bit_ctrl
DESIGNS["i2c_master_bit_ctrl.v"] = """module i2c_master_bit_ctrl (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [2:0] cmd,
    input  wire       din,
    input  wire       scl_i,
    input  wire       sda_i,
    output reg        scl_o,
    output reg        sda_o,
    output reg        dout,
    output reg        cmd_ack
);
    reg [2:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scl_o   <= 1'b1;
            sda_o   <= 1'b1;
            dout    <= 1'b0;
            cmd_ack <= 1'b0;
            state   <= 3'd0;
        end else begin
            case (state)
                3'd0: begin
                    cmd_ack <= 1'b0;
                    if (cmd == 3'd0) begin
                        sda_o <= 1'b0;
                        state <= 3'd1;
                    end else if (cmd == 3'd1) begin
                        scl_o <= 1'b1;
                        sda_o <= 1'b0;
                        state <= 3'd2;
                    end else if (cmd == 3'd3) begin
                        scl_o <= 1'b0;
                        sda_o <= din;
                        state <= 3'd3;
                    end else if (cmd == 3'd2) begin
                        scl_o <= 1'b0;
                        sda_o <= 1'b1;
                        state <= 3'd4;
                    end
                end
                3'd1: begin scl_o <= 1'b0; cmd_ack <= 1'b1; state <= 3'd0; end
                3'd2: begin sda_o <= 1'b1; cmd_ack <= 1'b1; state <= 3'd0; end
                3'd3: begin scl_o <= 1'b1; cmd_ack <= 1'b1; state <= 3'd0; end
                3'd4: begin scl_o <= 1'b1; dout <= sda_i; cmd_ack <= 1'b1; state <= 3'd0; end
            endcase
        end
    end
endmodule
"""

# 37. fifo_sync_8x8
DESIGNS["fifo_sync_8x8.v"] = """module fifo_sync_8x8 (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       wr_en,
    input  wire [7:0] wr_data,
    input  wire       rd_en,
    output reg  [7:0] rd_data,
    output wire       full,
    output wire       empty
);
    reg [7:0] mem [0:7];
    reg [2:0] wr_ptr, rd_ptr;
    reg [3:0] count;

    assign full  = (count == 4'd8);
    assign empty = (count == 4'd0);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr  <= 3'd0;
            rd_ptr  <= 3'd0;
            count   <= 4'd0;
            rd_data <= 8'b0;
        end else begin
            if (wr_en && !full) begin
                mem[wr_ptr] <= wr_data;
                wr_ptr <= wr_ptr + 1'b1;
            end
            if (rd_en && !empty) begin
                rd_data <= mem[rd_ptr];
                rd_ptr  <= rd_ptr + 1'b1;
            end
            case ({wr_en && !full, rd_en && !empty})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count;
            endcase
        end
    end
endmodule
"""

# 38. lifo_stack_8x8
DESIGNS["lifo_stack_8x8.v"] = """module lifo_stack_8x8 (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       push,
    input  wire       pop,
    input  wire [7:0] data_in,
    output reg  [7:0] data_out,
    output wire       empty,
    output wire       full
);
    reg [7:0] stack_mem [0:7];
    reg [3:0] sp;

    assign empty = (sp == 4'd0);
    assign full  = (sp == 4'd8);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sp       <= 4'd0;
            data_out <= 8'b0;
        end else begin
            if (push && !full) begin
                stack_mem[sp[2:0]] <= data_in;
                sp <= sp + 1'b1;
            end else if (pop && !empty) begin
                data_out <= stack_mem[sp[2:0] - 1'b1];
                sp <= sp - 1'b1;
            end
        end
    end
endmodule
"""

# 39. register_file_8x8
DESIGNS["register_file_8x8.v"] = """module register_file_8x8 (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       wr_en,
    input  wire [2:0] wr_addr,
    input  wire [7:0] wr_data,
    input  wire [2:0] rd_addr_a,
    output reg  [7:0] rd_data_a,
    input  wire [2:0] rd_addr_b,
    output reg  [7:0] rd_data_b
);
    reg [7:0] regs [0:7];
    integer k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (k = 0; k < 8; k = k + 1)
                regs[k] <= 8'b0;
            rd_data_a <= 8'b0;
            rd_data_b <= 8'b0;
        end else begin
            if (wr_en) regs[wr_addr] <= wr_data;
            rd_data_a <= regs[rd_addr_a];
            rd_data_b <= regs[rd_addr_b];
        end
    end
endmodule
"""

# 40. pwm_generator_8bit
DESIGNS["pwm_generator_8bit.v"] = """module pwm_generator_8bit (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] duty,
    input  wire [7:0] period,
    output reg        pwm_out
);
    reg [7:0] counter;
    reg [7:0] duty_r, period_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter  <= 8'b0;
            duty_r   <= 8'b0;
            period_r <= 8'b0;
            pwm_out  <= 1'b0;
        end else begin
            duty_r   <= duty;
            period_r <= period;
            if (counter >= period_r) counter <= 8'b0;
            else counter <= counter + 1'b1;
            pwm_out <= (counter < duty_r);
        end
    end
endmodule
"""


def generate_all(output_directory: str | Path = "data/raw_rtl") -> list[Path]:
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for filename, code in DESIGNS.items():
        dest = target / filename
        dest.write_text(code.strip() + "\n", encoding="utf-8")
        created.append(dest)
    return created


if __name__ == "__main__":
    files = generate_all()
    print(f"Successfully generated {len(files)} RTL designs in data/raw_rtl")

