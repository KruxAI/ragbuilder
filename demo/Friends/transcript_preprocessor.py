from bs4 import BeautifulSoup
import re
from pathlib import Path
import logging
import chardet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptCleaner:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Clean individual text segments."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove any remaining HTML entities
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&nbsp;', ' ', text)
        return text.strip()
    
    def get_specified_encoding(self, file_path: Path) -> str | None:
        """Extract charset from meta tag if specified."""
        try:
            # First read a small portion of the file to check for meta charset
            with open(file_path, 'rb') as f:
                # Read first 1024 bytes which typically contain the head section
                raw_content = f.read(1024)
                
            # Look for charset in meta tag in raw bytes
            charset_match = re.search(br'charset=([\w-]+)', raw_content, re.IGNORECASE)
            if charset_match:
                return charset_match.group(1).decode('ascii').lower()
            
            return None
                
        except Exception as e:
            logger.warning(f"Error reading charset from {file_path.name}: {str(e)}")
            return None

    def get_file_encoding(self, file_path: Path) -> str:
        """Get file encoding, first from meta tag, then using chardet."""
        # Try to get specified encoding first
        specified_encoding = self.get_specified_encoding(file_path)
        if specified_encoding:
            logger.info(f"Using specified encoding for {file_path.name}: {specified_encoding}")
            return specified_encoding
        
        # Fallback to chardet
        logger.info(f"No encoding specified in {file_path.name}, detecting encoding...")
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        logger.info(f"Detected encoding for {file_path.name}: {encoding} (confidence: {confidence:.2f})")
        return encoding
    
    def process_file(self, file_path: Path) -> str:
        """Process single transcript file."""
        encoding = self.get_file_encoding(file_path)

        with open(file_path, 'r', encoding=encoding) as f:
            soup = BeautifulSoup(f, 'html.parser')
            
        # Extract title
        title = soup.title.string if soup.title else file_path.stem
        cleaned_content = [f"Episode: {title}\n\n"]
        
        # Process all paragraphs
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if not text:
                continue
                
            # Skip transcriber information
            if any(skip in text.lower() for skip in ['transcribed by:', 'written by:']):
                continue
            
            # Clean and format the text
            text = self.clean_text(text)
            
            # Handle scene descriptions
            if text.startswith('[') and text.endswith(']'):
                cleaned_content.append(f"\n{text}\n")
            
            # Handle dialogue
            elif ':' in text:
                speaker, dialogue = text.split(':', 1)
                cleaned_content.append(f"{speaker.strip()}: {dialogue.strip()}")
            
            # Handle other content
            else:
                cleaned_content.append(text)
        
        return '\n'.join(cleaned_content)
    
    def process_all_files(self):
        """Process all transcript files in the input directory."""
        for file_path in self.input_dir.glob('*.html'):
            try:
                logger.info(f"Processing {file_path.name}")
                cleaned_content = self.process_file(file_path)
                
                # Save cleaned content
                output_file = self.output_dir / f"{file_path.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                    
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")

if __name__ == "__main__":
    cleaner = TranscriptCleaner(
        input_dir="path/to/friends/transcripts",
        output_dir="path/to/output/cleaned_transcripts"
    )
    cleaner.process_all_files()