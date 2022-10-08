package main
import "encoding/hex"
import "crypto/sha1"
import "encoding/binary"
import "encoding/base64"
import "strings"
import "fmt"
import "os"

func main() {
	nid := calculateNID(os.Args[1])
	fmt.Println(nid)
}


func calculateNID(symbolName string) string {
	nidSuffixKey := "518D64A635DED8C1E6B039B1C3E55230"
	hashBytes := make([]byte, 8)
	suffix, _ := hex.DecodeString(nidSuffixKey)

	symbol := append([]byte(symbolName), suffix...)
	hash := sha1.Sum(symbol)

	// The order of the bytes has to be reversed. We can hack big endian to do this.
	binary.LittleEndian.PutUint64(hashBytes, binary.BigEndian.Uint64(hash[:8]))

	// The final NID is the hash bytes base64'd with the last '=' character removed
	nidHash := base64.StdEncoding.EncodeToString(hashBytes)
	nidHash = nidHash[0 : len(nidHash)-1]

	//  We also need to replace all forward slashes with dashes for encoding reasons
	nidHash = strings.Replace(nidHash, "/", "-", -1)

	return nidHash
}
