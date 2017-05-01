/**
 * Created by gustavoromito on 5/1/17.
 */
public class FuncaoAtivacao {

    enum Type {
        Sigmoide
    }

    private static Type mType = Type.Sigmoide;

    public FuncaoAtivacao(Type type) {
        mType = type;
    }

    public double execute(double sum) {
        switch (mType) {
            case Sigmoide:
                return funcaoSigmoide(sum);
        }
        return 0.0;
    }

    // Fx = Resultado da Funcao Exponencial
    public double derivate(double fx) {
        switch (mType) {
            case Sigmoide:
                return derivadaSigmoide(fx);
        }
        return 0.0;
    }

    /* FUNCAO DE ATIVACAO SIGMOIDE */
    private double funcaoSigmoide(double sum) {
        return 1.0 / (1.0 + Math.exp(-sum));
    }

    private double derivadaSigmoide(double fx) {
        return fx * (1.0 - fx);
    }
}
