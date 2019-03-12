` time scale 1ns / 1ps
module vrf_PID_test;
	// Inputs
	reg clk;
	reg rst_n;
	reg sin;
	reg rx_finish;
	reg [7:0] Vin;
	
	// Outputs
	wire [19:0] controlword_hex;
	wire data_en;
	wire sout;
	wire En;
	wire led2;
	wire trigf;
	wire ledoeo;
	wire DIR;
	// Instantiate the Unit Under Test (UUT)
	// led_test uut (
	// .clk(clk),
	// .rst_n(rst_n),
	// .led(led)
	// );
	
	sig_inp uut(

		.clk_core(clk),  // 10MHz
		.rstn(rst_n),	
		.sin(sin),	//
		.rx_finish(rx_finish),	//
		.Vin(Vin), //ADC获取的数据

		.controlword_hex(controlword_hex),
		.data_en(data_en),
		.sout(sout),
		.En(En),// Temperature Control//		output reg IGBT_cp,
		.led2(led2),   //稳定控制点
		.trigf(trigf),//高电平 闭合 放大器电源连通
	    .ledoeo(ledoeo), //oeo稳定工作指示灯
		.DIR(DIR)
		  );
	integer i;
	
	initial begin
		// Initialize Inputs
		clk = 0;
		rst_n = 0;
		sin = 0;
		rx_finish = 0;
		Vin = 0;
		// Wait 100 ns for global reset to finish
		#100;
		rst_n = 1;
		// Add stimulus here
		for(i=0;i<=255;i=i+1)
		   begin
			#1000;
			Vin = i;
		   end
		   
		 for(i=255;i>=0;i=i-1)
		   begin
		   #1000;
			Vin = i;
		   end
		#200000000;
		$stop;
	end
	

	always #10 clk = ~ clk; //产生 50MHz 时钟源

	
	always @(posedge data_en)   
		begin
			rx_finish= 1;
			#100;
			rx_finish= 0;
		end
endmodule