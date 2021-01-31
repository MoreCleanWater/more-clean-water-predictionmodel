import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class CSV {
	
	public static void main(String args[])  
	{  
		FileWriter writer;
		try {
			writer = new FileWriter("C:\\Users\\researcher\\Dropbox\\MultiQ\\AWS\\events.csv"); // give a path here if needed
			
			List<String> dataToFile = new ArrayList<>();
			// columns, give names from your data
			dataToFile.add("col1");
			dataToFile.add("col2");
			dataToFile.add("col3");
			dataToFile.add("col4");
			// writing columns line to file
			writer.write(dataToFile.stream().collect(Collectors.joining(",")));
			writer.write("\n"); // newline
			// data
			dataToFile.clear(); // empty columns line
			int index = 0;
			int totalSize = 100;
			while (index < totalSize) {
				dataToFile.add("val1");
				dataToFile.add("val2");
				dataToFile.add("val3");
				dataToFile.add("val4");
				// writing columns line to file
				writer.write(dataToFile.stream().collect(Collectors.joining(",")));
				writer.write("\n"); // newline
				dataToFile.clear();
				index++;
			}
			
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}  
}
