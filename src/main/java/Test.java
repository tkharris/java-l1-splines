import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;
import java.util.stream.IntStream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import com.google.common.base.Function;


public class Test {
    public static void main(String... args) {
	try {
	    INDArray y = parseCRUTEM3("annual.txt");
	    LineChart_AWT temp_chart =
		new LineChart_AWT("Global Temps",
				  "Global Annual Average Temps",
				  IntStream.range(1850, 1850 + y.length()).toArray());
	    temp_chart.addSeries("Raw Average Temps", y.data().dup().asDouble());
	    double s = 200.0; 
	    INDArray y_hat = l1Spline_1d(y, s);
	    temp_chart.addSeries("L1 Spline", y_hat.data().dup().asDouble());
	    temp_chart.build();
	    temp_chart.saveChart("crutem3.png");
	    temp_chart.pack();
	    temp_chart.setVisible(true);
	} catch (Exception e) {
	    e.printStackTrace();
	}
    }

    /*
     * 1-D DCT (DCT-II)
     *
     * Fairly naive implementation of DCT. The cosine matrix could be memoized.
     * Also, the entire matrix is allocated into memory at once, which makes
     * the conputations rapid but limits the size of the problems that can be
     * addressed.
     */
    public static INDArray dct_1d(INDArray X) {
	final int N = X.length();

	INDArray counter = Nd4j.linspace(0, N-1, N)
	    .transpose()
	    .mmul(Nd4j.linspace(0, N-1, N).addi(0.5))
	    .muli(Math.PI/N);

	INDArray ret = Transforms.cos(counter, false)
	    .muliRowVector(X)
	    .sum(1)
	    .transpose()
	    .reshape(N);

	/**

	   We formulate 
	   <math>X_k =
 \sum_{n=0}^{N-1} x_n \cos \left[\frac{\pi}{N} \left(n+\frac{1}{2}\right) k \right] \quad \quad k = 0, \dots, N-1.
 </math>

 with vectors as:

 <math>\cos \left[\frac{\pi}{N} \left(\vec{n}+\fraq{1}{2}\right)^T \vec{k}] \vec{x}</math> yielding

<math>\begin{bmatrix}\cos \left(\fraq{\pi}{N} \left(n_0+\fraq{1}{2}\right) k_0 \right) x_0 & ... & \cos \left(\fraq{\pi}{N} \left(n_0+\fraq{1}{2}\right) k_{N-1} \right) x_0 \\
... & ... & ... \\
\cos \left(\fraq{\pi}{N} \left(n_{N-1}+\fraq{1}{2}\right) k_0 \right) x_{N-1} & ... & \cos \left(\fraq{\pi}{N} \left(n_{N-1}+\fraq{1}{2}\right) k_{N-1} \right) x_{N-1} \end{bmatrix}</math>

and then sum the columns.

	**/

	return ret;
    }

    /*
     * 1-D inverse DCT-II
     * 
     * Same caveots as above, could be memoized, inefficient use of memory, etc.
     */
    public static INDArray dct_inv_1d(INDArray X) {
	final int N = X.length();

	INDArray counter = Nd4j.linspace(0, N-1, N)
	    .addi(0.5)
	    .transpose()
	    .mmul(Nd4j.linspace(1, N-1, N-1))
	    .muli(Math.PI/N);

	INDArray ret = Transforms.cos(counter, false)
	    .muliRowVector(X.get(NDArrayIndex.interval(1, N)))
	    .sum(1)
	    .reshape(N)
	    .addi(X.getDouble(0)/2)
	    .muli(2.0/N);

	return ret;
    }

    /*
     * 1-D L1 regularization
     */
    public static INDArray shrink_1d(INDArray v, double gamma) {
	final int N = v.length();

	INDArray q = Transforms.max(Nd4j.zeros(N),
				    Transforms.abs(v).subi(gamma),
				    false)
	    .muli(Transforms.sign(v));
	return q;
    }

    /*
     * Convenience function to support reasonable defaults for liSpline
     */ 
    public static INDArray l1Spline_1d(INDArray y, double s) {
	return l1Spline_1d(y, s, 1.0, 1e-3, 1);
    }

    /*
     * 1-D L1 spline function. 
     */
    public static INDArray l1Spline_1d(INDArray y, double s, double lambda, double epsilon, int N_i) {
	final int N = y.length();

	INDArray Lambda = Nd4j.linspace(0, N-1, N)
	    .reshape(N)
	    .muli(Math.PI/N);
	Transforms.cos(Lambda, false);
	Lambda.muli(2).subi(2);

	INDArray Gamma = Nd4j.ones(N).reshape(N)
	    .diviRowVector(Lambda.mulRowVector(Lambda).muli(s).addi(1));

	INDArray d = Nd4j.zeros(N).reshape(N);
	INDArray b = Nd4j.zeros(N).reshape(N);
	INDArray z = null;

	int iter_max = 100;
	for(int k = 0; k < iter_max; k++) {
	    INDArray z_next = null;
	    for (int i=0; i < N_i; i++) {
		d.addiRowVector(y).subiRowVector(b);
		z_next = dct_inv_1d(Gamma.mulRowVector(dct_1d(d)));
		d = shrink_1d(z_next.subRowVector(y).addRowVector(b), 1.0/lambda);
	    }
	    if (z != null) {
		Double delta = (Double)z_next.subRowVector(z).norm2Number() / (Double)z.norm2Number();
		System.out.println(k + ": delta = " + delta);
		if (delta < epsilon) break;
	    }
	    b.addiRowVector(z_next).subiRowVector(y).subiRowVector(d);
	    z = z_next;
	}
	
	return z;
    }
    
    /*
     * Multi-dimentional L1 Spline
     * (Unfinished work in progress - Do not use)
     */
    public static INDArray l1Spline(INDArray y, double s, double lambda, double epsilon, int N_i) {
	final int m = y.rank();
	final int[] shape = y.shape();

	INDArray d = Nd4j.zeros(shape).reshape(shape);
	INDArray b = Nd4j.zeros(shape).reshape(shape);
	int iter_max = 100;
	
	do {
	    iter_max--;
	} while (iter_max != 0);

	return y;
    }

    /*
     * Loads CRUTEM3 data from file source
     */
    public static INDArray parseCRUTEM3(String fn) throws IOException {
	INDArray t;
	try (Stream<String> stream = Files.lines(Paths.get(fn))) {
	    double[] f_a = stream.map(line -> line.split("\\s+")[1])
		.mapToDouble(Double::parseDouble)
		.toArray();
	    t = Nd4j.create(f_a, new int[]{f_a.length});
	}
	return t;
    }
}
