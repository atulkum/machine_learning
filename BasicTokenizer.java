import com.google.common.base.Ascii;
import com.google.common.collect.Iterables;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/*
Reference: https://github.com/google-research/bert/blob/master/tokenization.py#L185
 */
public final class BasicTokenizer {
    private final boolean doLowerCase;

    public BasicTokenizer(boolean doLowerCase) {
        this.doLowerCase = doLowerCase;
    }

    public List<String> tokenize(String text) {
        String cleanedText = cleanText(text);

        List<String> origTokens = whitespaceTokenize(cleanedText);

        StringBuilder stringBuilder = new StringBuilder();
        for (String token : origTokens) {
            if (doLowerCase) {
                token = Ascii.toLowerCase(token);
            }
            List<String> list = runSplitOnPunc(token);
            for (String subToken : list) {
                stringBuilder.append(subToken).append(" ");
            }
        }
        return whitespaceTokenize(stringBuilder.toString());
    }

    /* Performs invalid character removal and whitespace cleanup on text. */
    static String cleanText(String text) {
        if (text == null) {
            throw new NullPointerException("The input String is null.");
        }

        StringBuilder stringBuilder = new StringBuilder("");
        for (int index = 0; index < text.length(); index++) {
            char ch = text.charAt(index);

            // Skip the characters that cannot be used.
            if (isInvalid(ch) || isControl(ch)) {
                continue;
            }
            if (isWhitespace(ch)) {
                stringBuilder.append(" ");
            } else {
                stringBuilder.append(ch);
            }
        }
        return stringBuilder.toString();
    }

    /* Runs basic whitespace cleaning and splitting on a piece of text. */
    static List<String> whitespaceTokenize(String text) {
        if (text == null) {
            throw new NullPointerException("The input String is null.");
        }
        return Arrays.asList(text.split(" "));
    }

    /* Splits punctuation on a piece of text. */
    static List<String> runSplitOnPunc(String text) {
        if (text == null) {
            throw new NullPointerException("The input String is null.");
        }

        List<String> tokens = new ArrayList<>();
        boolean startNewWord = true;
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (isPunctuation(ch)) {
                tokens.add(String.valueOf(ch));
                startNewWord = true;
            } else {
                if (startNewWord) {
                    tokens.add("");
                    startNewWord = false;
                }
                tokens.set(tokens.size() - 1, Iterables.getLast(tokens) + ch);
            }
        }

        return tokens;
    }

    public static boolean isInvalid(char ch) {
        return (ch == 0 || ch == 0xfffd);
    }

    /** To judge whether it's a control character(exclude whitespace). */
    public static boolean isControl(char ch) {
        if (Character.isWhitespace(ch)) {
            return false;
        }
        int type = Character.getType(ch);
        return (type == Character.CONTROL || type == Character.FORMAT);
    }

    /** To judge whether it can be regarded as a whitespace. */
    public static boolean isWhitespace(char ch) {
        if (Character.isWhitespace(ch)) {
            return true;
        }
        int type = Character.getType(ch);
        return (type == Character.SPACE_SEPARATOR
                || type == Character.LINE_SEPARATOR
                || type == Character.PARAGRAPH_SEPARATOR);
    }

    /** To judge whether it's a punctuation. */
    public static boolean isPunctuation(char ch) {
        int type = Character.getType(ch);
        return (type == Character.CONNECTOR_PUNCTUATION
                || type == Character.DASH_PUNCTUATION
                || type == Character.START_PUNCTUATION
                || type == Character.END_PUNCTUATION
                || type == Character.INITIAL_QUOTE_PUNCTUATION
                || type == Character.FINAL_QUOTE_PUNCTUATION
                || type == Character.OTHER_PUNCTUATION);
    }
}